# Defined in ConScape.jl for load order
# abstract type Solver end
@doc """
    Solver

Abstract supertype for ConScape solvers.
""" Solver 

function init(s::Solver, 
    cm::FundamentalMeasure, 
    p::AbstractProblem, 
    g::Grid
) 
    Pref = _Pref(g.affinities)
    W = _W(Pref, cm.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse rhs
    B_sparse = sparse_rhs(g.targetnodes, size(g.costmatrix, 1))
    A_init = init(s, A)
    B_dense = Matrix(B_sparse)
    workspace = copy(B_dense)
    Z = ldiv!(s, A_init, B_dense; B_copy=workspace)
    # Check that values in Z are not too small:
    _check_z(s, Z, W, g)
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    return _measures_workspace(p, grsp; A, A_init, workspace, B_sparse)
end

# RSP is not used for ConnectivityMeasure, so the solver isn't used
function solve(s::Solver, cm::ConnectivityMeasure, p::AbstractProblem, g::Grid;
    workspace=init(s, cm, p, g),
) 
    return map(p.graph_measures) do gm
        compute(gm, p, g; solver=s, workspace...)
    end
end
function solve(s::Solver, cm::FundamentalMeasure, p::AbstractProblem, g::Grid; 
    workspace=nothing,
) 
    workspace = isnothing(workspace) ? init(s, cm, p, g) : workspace
    # TODO remove use of GridRSP where possible
    results = map(p.graph_measures) do gm
        compute(gm, p, workspace.grsp; workspace...)
    end
    return _merge_to_stack(results)
end

function init(s::Solver, cm::ConnectivityMeasure, p::AbstractProblem, g::Grid) 
    # TODO what is needed here?
    return (;)
end

LinearAlgebra.ldiv!(solver::Solver, A, B; kw...) = ldiv!(solver, init(solver, A), A, B; kw...)

"""
   MatrixSolver(; check)

Solve all operations on a fully materialised Z matrix.

This is fast but memory inneficient for CPUS, and isn't threaded.
But may be best for GPUs using CuSSP.jl ?
"""
@kwdef struct MatrixSolver <: Solver 
    check::Bool = true
end

init(::Union{Nothing,MatrixSolver}, A::AbstractMatrix) = (; F=lu(A))

# TODO: no type pyracy
LinearAlgebra.ldiv!(::Union{MatrixSolver,Nothing}, (; F), B; B_copy=copy(B)) = 
    ldiv!(B, F, B_copy)

"""
   VectorSolver(; check, threaded)

Use julias default solver but broken into columns, with 
less memory use and the capacity for threading
"""
@kwdef struct VectorSolver <: Solver 
    check::Bool = true
    threaded::Bool = false
end

function init(s::VectorSolver, A::AbstractMatrix)
    F = lu(A)
    Tb = Vector{eltype(A)}
    if s.threaded
        nbuffers = Threads.nthreads()
        # channel = Channel{Tuple{typeof(F),Vector{Float64}}}(nbuffers)
        # Create one init per thread
        # UMFPACK `copy` shares memory but avoids workspace race conditions
        [
            (; 
                F=(i == 1 ? F : copy(F)), 
                b=Tb(undef, size(A, 2))
            )
            for i in 1:nbuffers
        ]
    else
        b = Tb(undef, size(A, 2))
        return [(; F, b)]
    end
end

function LinearAlgebra.ldiv!(s::VectorSolver, init, B; B_copy=nothing)
    transposeoptype = SparseArrays.LibSuiteSparse.UMFPACK_A
    # for SparseArrays.UMFPACK._AqldivB_kernel!(Z, F, B, transposeoptype)

    # This is basically SparseArrays.UMFPACK._AqldivB_kernel!
    # But we unroll it to avoid copies or allocation of B
    if s.threaded
        channel = Channel{typeof(init[1])}(length(init))
        for x in init
            put!(channel, x)
        end
        # Create a channel to store problem b vectors for threads
        # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
        Threads.@threads for col in 1:size(B, 2)
            # Get a workspace from the channel
            F_t, b_t = take!(channel)
            # Copy a column from B
            b_t .= view(B, :, col)
            # Solve for the column
            SparseArrays.UMFPACK.solve!(view(B, :, col), F_t, b_t, transposeoptype)
            # Reuse the workspace 
            put!(channel, (F_t, b_t))
        end
    else
        (; F, b) = init[1]
        for col in 1:size(B, 2)
            b .= view(B, :, col)
            SparseArrays.UMFPACK.solve!(view(B, :, col), F, b, transposeoptype)
        end
    end

    return B
end

"""
   LinearSolver(args...; threded, kw...)

Solve all operations column-by-column using LinearSolve.jl solvers.

The `threaded` keyword specifies if threads are used per target.
Other arguments and keywords are passed to `LinearSolve.solve` after the
problem object, like:

````julia
`LinearSolve.solve(linearproblem, args...; kw...)`
````

# Example

This example uses LinearSolve.jl wth `KrylovJL_GMRES` and a preconditioner.

TODO: an example that is realistic

````julia
using LinearSolve
problem = ConScape.Problem(; 
    solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
    graph_measures = (;
        func=ConScape.ConnectedHabitat(),
        qbetw=ConScape.BetweennessQweighted(),
    ),
    distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor()),
    connectivity_measure = ConScape.ExpectedCost(θ=1.0),
)
````
"""
struct LinearSolver <: Solver 
    args
    keywords
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

function LinearAlgebra.ldiv!(s::LinearSolver, (; linsolve, channel, b), B)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    if s.threaded
        Threads.@threads for i in 1:size(B, 2)
            # Get column memory from the channel
            linsolve_t, b_t = take!(channel)
            # Update it
            b_t .= view(B, :, i)
            # Update solver with new b values
            reinit!(linsolve_t; b=b_t, reuse_precs=false)
            sol = LinearSolve.solve(linsolve_t, s.args...; s.keywords...)
            # Aim for something like this ?
            # res = map(connectivity_measures(p)) do cm
            #     compute(cm, g, sol.u, i)
            # end
            # For now just use Z
            B[:, i] .= sol.u
            put!(channel, (linsolve_t, b_t))
        end
    else
        for i in 1:size(B, 2)
            b .= view(B, :, i)
            reinit!(linsolve; b, reuse_precs=true)
            sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
            # Udate the column
            B[:, i] .= sol.u
        end
    end
    return B
end

function init(s::LinearSolver, A)
    b = zeros(eltype(A), size(A, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, s.args...; s.keywords...)
    # TODO what is needed here?
    nbuffers = Threads.nthreads()
    # Create a channel to store problem b vectors for threads
    # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    channel = Channel{Tuple{typeof(linsolve),Vector{Float64}}}(nbuffers)
    for i in 1:nbuffers
        # TODO fix this in LinearSolve.jl with batching
        # We should not need to `deepcopy` the whole problem we 
        # just need to replicate the specific workspace arrays 
        # that will cause race conditions.
        # But currently there is no parallel mode for LinearSolve.jl
        # See https://github.com/SciML/LinearSolve.jl/issues/552
        put!(channel, (deepcopy(linsolve), Vector{eltype(A)}(undef, size(A, 2))))
    end
    return (; linsolve, channel, b)
end

# Utils

# We may have multiple distance_measures per
# graph_measure, but we want a single RasterStack.
# So we merge the names of the two layers

function _merge_to_stack(nt::NamedTuple{K}) where K
    unique_nts = map(K) do k
        _mergename(Val{k}(), nt[k])
    end
    # merge unique layers into a sinlge RasterStack
    nt = merge(unique_nts...)
    if all(map(x -> x isa Raster, nt))
        return RasterStack(nt)
    else
        return nt # Cant return a RasterStack for these outputs 
    end
end
_maybe_raster(x::Raster) = x
_maybe_raster(x::Number) = Raster(fill(x), ())
_maybe_raster(x) = x

function _mergename(::Val{K1}, gm::NamedTuple{K2}) where {K1, K2}
    # Combine outer and inner names with an underscore
    joinedkeys = map(K2) do k2
        Symbol(K1, :_, k2)
    end
    # And rename the NamedTuple
    NamedTuple{joinedkeys}(map(_maybe_raster, values(gm)))
end
_mergename(::Val{K1}, gm) where {K1, K2} =
    # We keep the name as is
    NamedTuple{(K1,)}((_maybe_raster(gm),))

function _check_z(s, Z, W, g)
    # Check that values in Z are not too small:
    if hasproperty(s, :check) && s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end

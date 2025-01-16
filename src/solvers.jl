# Defined in ConScape.jl for load order
# abstract type Solver end
@doc """
    Solver

Abstract supertype for ConScape solvers.
""" Solver 

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

# Fallback generic ldiv solver
solve_ldiv!(solver, A, B) = 
    solve_ldiv!(solver, init(solver, A), A, B)

"""
   MatrixSolver(; check)

Solve all operations on a fully materialised Z matrix.

This is fast but memory inneficient for CPUS, and isn't threaded.
But may be best for GPUs using CuSSP.jl ?
"""
@kwdef struct MatrixSolver <: Solver 
    check::Bool = true
end

solve_ldiv!(::Union{MatrixSolver,Nothing}, (; F), A, B) = ldiv!(F, B)

init(::Union{Nothing,MatrixSolver}, A::AbstractMatrix) = (; F=lu(A))

"""
   VectorSolver(; check, threaded)

Use julias default solver but broken into columns, with 
less memory use and the capacity for threading
"""
@kwdef struct VectorSolver <: Solver 
    check::Bool = true
    threaded::Bool = false
end

function init(s::Union{MatrixSolver,VectorSolver,Nothing}, 
    cm::FundamentalMeasure, 
    p::AbstractProblem, 
    g::Grid
) 
    (; A, B, Pref, W) = setup_sparse_problem(g, cm)
    # Check that values in Z are not too small:
    A_init = init(s, A)
    Z = solve_ldiv!(s, A_init, A, Matrix(B))
    _check_z(s, Z, W, g)
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    return (; grsp, _measures_workspace(p, grsp; A, A_init)...)
end

function init(s::VectorSolver, A::AbstractMatrix)
    F = lu(A)
    if s.threaded
        nbuffers = Threads.nthreads()
        channel = Channel{Tuple{typeof(F),Vector{Float64}}}(nbuffers)
        for _ in 1:nbuffers
            # TODO not all of F needs to be duplicated?
            # Can we just copy the workspace arrays and resuse the rest?
            put!(channel, (deepcopy(F), Vector{eltype(A)}(undef, size(A, 2))))
        end
        return (; F, channel)
    else
        b = zeros(eltype(A), size(A, 2))
        return (; F, b)
    end
end

function solve_ldiv!(s::VectorSolver, init, A, B)
    transposeoptype = SparseArrays.LibSuiteSparse.UMFPACK_A
    # for SparseArrays.UMFPACK._AqldivB_kernel!(Z, F, B, transposeoptype)

    # This is basically SparseArrays.UMFPACK._AqldivB_kernel!
    # But we unroll it to avoid copies or allocation of B
    if s.threaded
        (; F, channel) = init
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
        (; F, b) = init
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

function solve_ldiv!(s::LinearSolver, A, B)
    b = zeros(eltype(A), size(B, 1))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, s.args...; s.keywords...)
    solve_ldiv!(s, linsolve, A, B)
end
function solve_ldiv!(s::LinearSolver, linsolve, A, B)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    # if s.threaded
    #     nbuffers = Threads.nthreads()
    #     # Create a channel to store problem b vectors for threads
    #     # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    #     ch = Channel{Tuple{typeof(linsolve),Vector{Float64}}}(nbuffers)
    #     for i in 1:nbuffers
    #         # TODO fix this in LinearSolve.jl with batching
    #         # We should not need to `deepcopy` the whole problem we 
    #         # just need to replicate the specific workspace arrays 
    #         # that will cause race conditions.
    #         # But currently there is no parallel mode for LinearSolve.jl
    #         # See https://github.com/SciML/LinearSolve.jl/issues/552
    #         put!(ch, (deepcopy(linsolve), Vector{eltype(A)}(undef, size(B, 1))))
    #     end
    #     Threads.@threads for i in 1:size(B, 2)
    #         # Get column memory from the channel
    #         linsolve_t, b_t = take!(ch)
    #         # Update it
    #         b_t .= view(B, :, i)
    #         # Update solver with new b values
    #         reinit!(linsolve_t; b=b_t, reuse_precs=true)
    #         sol = LinearSolve.solve(linsolve_t, s.args...; s.keywords...)
    #         # Aim for something like this ?
    #         # res = map(connectivity_measures(p)) do cm
    #         #     compute(cm, g, sol.u, i)
    #         # end
    #         # For now just use Z
    #         B[:, i] .= sol.u
    #         put!(ch, (linsolve_t, b_t))
    #     end
    # else
        for i in 1:size(B, 2)
            b .= view(B, :, i)
            reinit!(linsolve; b, reuse_precs=true)
            sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
            # Udate the column
            B[:, i] .= sol.u
        end
    # end
    return B
end


# Utils

function setup_sparse_problem(g::Grid, cm::FundamentalMeasure)
    Pref = _Pref(g.affinities)
    W = _W(Pref, cm.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse rhs
    B = sparse_rhs(g.targetnodes, size(g.costmatrix, 1))
    return (; A, B, Pref, W)
end

# We may have multiple distance_measures per
# graph_measure, but we want a single RasterStack.
# So we merge the names of the two layers

function _merge_to_stack(nt::NamedTuple{K}) where K
    unique_nts = map(K) do k
        gm = nt[k]
        if gm isa NamedTuple
            # Combine outer and inner names with an underscore
            joinedkeys = map(keys(gm)) do k_inner
                Symbol(k, :_, k_inner)
            end
            # And rename the NamedTuple
            NamedTuple{joinedkeys}(map(_maybe_raster, values(gm)))
        else
            # We keep the name as is
            NamedTuple{(k,)}((_maybe_raster(gm),))
        end
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

function _check_z(s, Z, W, g)
    # Check that values in Z are not too small:
    if s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end
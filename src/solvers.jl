# Defined in ConScape.jl for load order
# abstract type Solver end
function init!(
    ws::NamedTuple, 
    s::Solver, 
    cm::FundamentalMeasure, 
    p::AbstractProblem, 
    rast::RasterStack,
) 
    grid = g = Grid(p, rast)
    gms = graph_measures(p)
    cf = connectivity_function(p)
    Pref = _Pref(g.affinities)
    W = _W(Pref, cm.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse rhs
    B_sparse = sparse_rhs(g.targetnodes, size(g.costmatrix, 1))
    # A_init = haskey(ws, :A_init) ? init(s, A) : init!(ws.A_init, s, A)
    A_init = init(s, A)
    # B_dense becomes Z
    B_dense = haskey(ws, :Z) ? copyto!(_resize(ws.Z, size(B_sparse)), B_sparse) : Matrix(B_sparse)
    n_workspaces = count_workspaces(p)
    n_permuted_workspaces = count_permuted_workspaces(p)
    # @show haskey(ws, :workspaces) 
    workspaces = if haskey(ws, :workspaces) 
        [_reshape(w, size(B_dense)) for w in ws.workspaces]
    else
        [similar(B_dense) for _ in 1:n_workspaces]
    end
    permuted_workspaces = if haskey(ws, :workspaces) 
        [_reshape(pw, size(B_dense')) for pw in ws.permuted_workspaces]
    else
        [similar(B_dense') for _ in 1:n_permuted_workspaces]
    end
    Z = ldiv!(s, A_init, B_dense; B_copy=copyto!(workspaces[1], B_dense))
    # Check that values in Z are not too small:
    _check_z(s, Z, W, g)
    grsp = GridRSP(grid, cm.θ, Pref, W, Z)

    Zⁱ = if hastrait(needs_inv, gms)
        haskey(ws, :Zⁱ) ? _inv!(_reshape(ws.Zⁱ, size(Z)), Z) : _inv(Z)
    else
        nothing
    end
    Aadj_init, Aadj = if hastrait(needs_Aaj_init, gms)
        # Just take the adjoint of the factorization of A
        # where possible to save calculations and memory
        Aadj_init, Aadj = if hasproperty(A_init, :F)
            Aadj = A'
            # Use adjoint factorization of A rather than recalculating for A'
            Aadj_init = merge(A_init, (; F=A_init.F'))
            Aadj_init, Aadj
        else
            # LinearSolve.jl cant handle the adjoint 
            # so we duplicate work and allocations
            Aadj = sparse(A')
            Aadj_init = init(solver(p), Aadj)
            Aadj_init, Aadj
        end
        Aadj_init, Aadj
    else
        nothing, nothing
    end
    # Create an intermediate workspace to use in computations
    workspace_kw = (; Zⁱ, workspaces, permuted_workspaces, Aadj_init, Aadj, A, A_init)
    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        ConScape.expected_cost(grsp; workspace_kw..., solver=solver(p))
    else
        nothing
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        ConScape.free_energy_distance(grsp; workspace_kw..., solver=solver(p))
    else
        nothing
    end
    proximities = if hastrait(needs_proximity, gms)
        # We populate this during `solve`
        haskey(ws, :proximities) ? _reshape(ws.proximities, size(Z)) : similar(Z)
    else
        nothing
    end

    # TODO make a trait
    CW = grsp.g.costmatrix .* grsp.W
    return (; grid, grsp, workspace_kw..., CW, free_energy_distances, expected_costs, proximities)
end

# RSP is not used for ConnectivityMeasure, so the solver isn't used
function solve!(
    workspace::NamedTuple, 
    s::Solver, 
    cm::ConnectivityMeasure, 
    p::AbstractProblem,
) 
    g = workspace.grid
    return map(p.graph_measures) do gm
        compute(gm, p, ; workspace...)
    end
end
function solve!(
    workspace::NamedTuple,
    s::Solver, 
    cm::FundamentalMeasure, 
    p::Problem,
) 
    g = workspace.grid
    gms = graph_measures(p)
    distance_transformation = cm.distance_transformation
    results = if distance_transformation isa NamedTuple
        # Map over both distance transformations and graph measures
        nested = map(distance_transformation) do dt
            cm1 = ConstructionBase.setproperties(cm, (; distance_transformation=dt))
            hastrait(needs_proximity, gms) &&
                _setproximities!(workspace.proximities, workspace.expected_costs, cm1, p, workspace.grsp)
            # Rebuild the problem with a connectivity measure
            # holding a single distance transformation, in case its used
            p1 = ConstructionBase.setproperties(p, (; connectivity_measure=cm1))
            map(gms) do gm
                if needs_connectivity(gm)
                    compute(gm, p1, workspace.grsp; workspace...)
                else
                    nothing
                end
            end
        end
        # Map over graph measures that don't need connectivity
        flat = map(gms) do gm
            if needs_connectivity(gm)
                nothing
            else
                compute(gm, p, workspace.grsp; workspace...)
            end
        end
        # Combine nested and flat results
        map(keys(gms)) do k
            f = flat[k]
            if isnothing(f) 
                map(n -> n[k], nested)
            else
                f
            end
        end |> NamedTuple{keys(gms)}
    else
        hastrait(needs_proximity, gms) &&
            _setproximities!(workspace.proximities, workspace.expected_costs, cm, p, workspace.grsp)
        # Map over graph measures
        map(p.graph_measures) do gm
            compute(gm, p, workspace.grsp; workspace...)
        end
    end
    return _merge_to_stack(results)
end

function init!(workspace::NamedTuple, s::Solver, cm::ConnectivityMeasure, p::AbstractProblem, rast::RasterStack) 
    # TODO what is needed here?
    return (; grid=Grid(p, rast))
end

LinearAlgebra.ldiv!(solver::Solver, A::AbstractMatrix, B::AbstractMatrix; kw...) = 
    ldiv!(solver, init(solver, A), B; kw...)

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
_mergename(::Val{K1}, gm) where K1 =
    # We keep the name as is
    NamedTuple{(K1,)}((_maybe_raster(gm),))

function _check_z(s, Z, W, g)
    # Check that values in Z are not too small:
    if hasproperty(s, :check) && s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end

# This duplicats some logic from gridrsp
function _setproximities!(
    proximities::AbstractMatrix, 
    expected_costs::AbstractMatrix, 
    cm::ConnectivityMeasure, 
    p::Problem,
    grsp::GridRSP
)
    g = grsp.g
    dt = cm.distance_transformation
    if isnothing(dt)
        dt = inv(g.costfunction)
    end
    map!(dt, proximities, expected_costs)
    _maybe_set_diagonal!(proximities, g, diagvalue(p))
    return proximities
end

function _reshape(A::Array, dims::Tuple{Vararg{Int}})
    len = prod(dims)
    mem = getfield(A, :ref).mem
    if size(A) == dims
        A
    elseif length(mem) >= len
        v = vec(A)
        # Hack to shrink the array
        setfield!(v, :size, (len,))
        reshape(v, dims)
    else
        v = resize!(vec(A), len)
        reshape(v, dims)
    end
end
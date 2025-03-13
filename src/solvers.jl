"""
   MatrixSolver(; check)

Solve all operations on a fully materialised Z matrix.

This is fast but memory inneficient for CPUS, and isn't threaded.
But may be best for GPUs using CuSSP.jl ?
"""
@kwdef struct MatrixSolver <: Solver
    check::Bool = true
end

"""
   VectorSolver(; check, threaded)

Use julias default solver but broken into columns, with 
less memory use and the capacity for threading
"""
@kwdef struct VectorSolver <: Solver
    check::Bool = true
    threaded::Bool = false
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
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor()),
problem = ConScape.Problem(; 
    solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
    graph_measures = (;
        func=ConScape.ConnectedHabitat(),
        qbetw=ConScape.BetweennessQweighted(),
    ),
    connectivity_measure = ConScape.ExpectedCost(θ=1.0),
)
````
"""
struct LinearSolver{A,K} <: Solver
    args::A
    keywords::K
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

# In `init!` we allocate all large dense arrays 
function init!(
    gp::GridPrecalculations,
    solver::Solver,
    p::Problem,
    rast::RasterStack;
    verbose=false,
)
    # Initialise the whole grid
    grid = Grid(p, rast; prune=false)
    # Initialise the workspace
    precalculate!(gp, grid; verbose)
end

# _init_dense! may be called multiple times from `init!`, for each thread
function _init_dense!(
    ws::NamedTuple,
    solver::Solver,
    cm::FundamentalMeasure,
    p::AbstractProblem,
    grid::Grid;
    verbose=false,
    reuse_output=false,
)
    verbose && println("Retreiving measures...")
    subgrids = split_subgraphs(grid)
    g = first(subgrids)
    gms = graph_measures(p)
    cf = connectivity_function(p)
    verbose && println("Defining sparse arrays...")
    # B_dense becomes Z
    verbose && println("Allocating workspaces...")
    sze = _workspace_size(solver, g)
    Z = if hastrait(needs_inv, gms)
        if haskey(ws, :Z)
            _reshape(ws.Z, sze)
        else
            Matrix{eltype(g.affinities)}(undef, sze)
        end
    else
        nothing
    end
    Zⁱ = if hastrait(needs_inv, gms)
        haskey(ws, :Zⁱ) ? _reshape(ws.Zⁱ, sze) : similar(Z)
    else
        nothing
    end
    n_workspaces = count_workspaces(p)
    n_permuted_workspaces = count_permuted_workspaces(p)
    workspaces = if haskey(ws, :workspaces)
        [_reshape(w, size(Z)) for w in ws.workspaces]
    else
        [similar(Z) for _ in 1:n_workspaces]
    end
    permuted_workspaces = if haskey(ws, :workspaces)
        [_reshape(pw, size(Z')) for pw in ws.permuted_workspaces]
    else
        [similar(Z') for _ in 1:n_permuted_workspaces]
    end
    # TODO these shouldn't have traits, it 
    # should be baked into the problem.
    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        haskey(ws, :expected_costs) ? _reshape(ws.expected_costs, size(Z)) : similar(Z)
    else
        nothing
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        haskey(ws, :free_energy_distances) ? _reshape(ws.free_energy_distances, size(Z)) : similar(Z)
    else
        nothing
    end
    proximities = if hastrait(needs_proximity, gms)
        haskey(ws, :proximities) ? _reshape(ws.proximities, size(Z)) : similar(Z)
    else
        nothing
    end
    # We don't re-use outputs
    outputs = if reuse_output && haskey(ws, :outputs)
        ws.outputs
    else
        if distance_transformation(cm) isa NamedTuple
            map(gms) do gm
                if needs_connectivity(gm)
                    map(distance_transformation(cm)) do dt
                        allocate_output(gm)
                    end
                else
                    allocate_output(gm)
                end
            end
        else
            map(gms) do gm
                allocate_output(gm)
            end
        end
    end

    verbose && println("Finished allocating...")

    return (; Z, Zⁱ, workspaces, permuted_workspaces, free_energy_distances, expected_costs, proximities, outputs, grid, subgrids)
end
# function init!(
#     workspace::NamedTuple, s::Solver, cm::ConnectivityMeasure, p::AbstractProblem, rast::RasterStack;
#     verbose=false,
#     grid=Grid(p, rast),
# )
#     # TODO what is needed here?
#     return (; grid)
# end

# RSP is not used for ConnectivityMeasure, so the solver isn't used
function solve!(
    workspace::NamedTuple,
    s::MatrixSolver,
    cm::ConnectivityMeasure,
    p::AbstractProblem;
    verbose=false
)
    g = workspace.g
    return map(graph_measures(p), workspace.outputs) do gm, output
        compute(gm, p; workspace..., output, verbose)
    end
end
function solve!(
    ws::NamedTuple,
    solver::MatrixSolver,
    cm::FundamentalMeasure,
    p::Problem;
    verbose=false,
)
    # Loop over unnconnected subgrids 
    sg1 = first(ws.subgrids)
    ws1 = _init_dense!(ws, solver, cm, p, sg1; verbose)
    for subgrid in ws.subgrids
        ws2 = _init_dense!(ws1, solver, cm, p, subgrid; verbose, reuse_output=true)
        ws3 = _init_sparse(ws2, solver, cm, p, subgrid; verbose)
        ws4 = _solve_dense!(ws3, solver, cm, p; verbose)
        gms = graph_measures(p)
        _solve!(ws4, solver, cm, cm.distance_transformation, gms, p; verbose)
    end
    @show typeof(ws1.outputs)
    return _merge_to_stack(_maybe_raster(ws1.outputs, sg1))
end
function solve!(
    ws::NamedTuple,
    solver::Union{VectorSolver,LinearSolver},
    cm,
    p::Problem;
    verbose=false,
)
    sg1 = first(ws.subgrids) 
    gms = graph_measures(p)
    # Predefine min-vectors targets (not worth putting in the workspace) ?
    targetnodes = sg1.targetnodes[1:1]
    targetidx = sg1.targetidx[1:1]
    qt = sg1.qt[1:1]
    target_allocs = (; targetidx, targetnodes, qt)
    target_grid = ConstructionBase.setproperties(sg1, target_allocs)
    # Allocate dense arrays at the single target size
    ws1 = _init_dense!(ws, solver, cm, p, target_grid; verbose)

    # Internally we solve one target at a time, for each prefactorized subgrid
    function solve_target!(workspace, subgrid, i)
        target_grid = ConstructionBase.setproperties(workspace.grid, target_allocs)
        _update_targets!(target_allocs, subgrid, i)
        # And rebuild the workspace with the new grid
        target_ws = (; workspace..., g=target_grid, grid=target_grid)
        target_ws1 = _solve_dense!(target_ws, solver, cm, p; verbose)
        _solve!(target_ws1, solver, cm, cm.distance_transformation, gms, p; verbose)
    end

    # Loop over unnconnected subgrids (there may be only one)
    for subgrid in ws.subgrids
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        ws2 = _init_sparse(ws1, solver, cm, p, subgrid; verbose)
        if isthreaded(solver)
            isthreaded(p) && error("threading at solver level not yet implemented")
            # Threads.@threads for i in eachindex(g.targetnodes)[2:end]
            # run(i)
            # end
        else
            # Then solve each target as a single right hand side column
            for i in eachindex(subgrid.targetnodes)
                solve_target!(ws2, subgrid, i)
            end
        end
    end
    return _merge_to_stack(_maybe_raster(ws1.outputs, ws.grid))
end

function _solve!(workspace, solver, cm, dt::NamedTuple{DT}, gms::NamedTuple{GMS}, p; verbose) where {DT,GMS}
    (; grid, Pref, W, Z, outputs) = workspace
    # GridRSP is just a wrapper now, we can remove it later
    grsp = GridRSP(grid, cm.θ, Pref, W, Z)
    # Map over both distance transformations and graph measures
    nested = map(values(dt), DT) do dt, k
        cm1 = ConstructionBase.setproperties(cm, (; distance_transformation=dt))
        hastrait(needs_proximity, gms) &&
            _setproximities!(workspace.proximities, workspace.expected_costs, cm1, p, grsp)
        # Rebuild the problem with a connectivity measure
        # holding a single distance transformation, in case its used
        p1 = ConstructionBase.setproperties(p, (; connectivity_measure=cm1))
        map(gms, outputs) do gm, os
            if needs_connectivity(gm)
                compute(gm, p1, grsp; workspace..., output=os[k])
            else
                nothing
            end
        end
    end |> NamedTuple{DT}
    # Map over graph measures that don't need connectivity
    flat = map(gms, outputs) do gm, output
        if needs_connectivity(gm)
            nothing
        else
            compute(gm, p, grsp; workspace..., output)
        end
    end
    # Combine nested and flat results
    return map(GMS) do k
        f = flat[k]
        if isnothing(f)
            map(n -> n[k], nested)
        else
            f
        end
    end |> NamedTuple{GMS}
end
function _solve!(workspace, solver, cm, dt, gms::NamedTuple{GMS}, p; verbose) where {GMS}
    (; grid, Pref, W, Z, outputs) = workspace
    # GridRSP is just a wrapper now, we can remove it later
    grsp = GridRSP(grid, cm.θ, Pref, W, Z)
    hastrait(needs_proximity, gms) &&
        _setproximities!(workspace.proximities, workspace.expected_costs, cm, p, grsp)
    # Map over graph measures
    map(p.graph_measures, outputs) do gm, output
        compute(gm, p, grsp; workspace..., output)
    end
end

function _update_targets!(a, g, i)
    # target_qualities[:, 1] = g.target_qualities[i]
    a.targetidx[1] = g.targetidx[i]
    a.targetnodes[1] = g.targetnodes[i]
    a.qt[1] = g.qt[i]
    return nothing
end

# Do all the work shared accross outputs
function _solve_dense!(ws::NamedTuple, solver::Solver, cm, p::Problem;
    verbose=false
)
    (; grid, W, Pref, A, A_init, Aadj_init, Aadj) = ws
    gms = graph_measures(p)
    cf = connectivity_function(p)
    # Sparse rhs
    # TODO get rid of this allocation
    # For VectorSolver we can write values directly to B
    B_sparse = sparse_rhs(grid.targetnodes, size(grid.costmatrix, 1))
    Z = if hastrait(needs_Z, gms)
        # verbose && 
        B = _reshape(ws.Z, size(B_sparse))
        copyto!(B, B_sparse)
        verbose && println("Solving Z matrix...")
        # Check that values in Z are not too small:

        Z = ldiv!(solver, A_init, B; B_copy=copyto!(ws.workspaces[1], B))
        # verbose && _check_z(s, Z, W, g)
        Z
    end
    Zⁱ = if hastrait(needs_inv, gms)
        verbose && println("Inverting Z...")
        _inv!(_reshape(ws.Zⁱ, size(Z)), Z)
    end

    grsp = GridRSP(grid, cm.θ, Pref, W, Z)
    workspace = (; ws..., Pref, W, A, A_init, Aadj, Aadj_init, Z, Zⁱ)

    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        verbose && println("Calculating expected cost...")
        expected_costs = _reshape(ws.expected_costs, size(Z))
        ConScape.expected_cost(grsp; workspace..., expected_costs, solver)
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        verbose && println("Calculating free energy distance...")
        free_energy_distances = _reshape(ws.free_energy_distances, size(Z))
        ConScape.free_energy_distance(grsp; workspace..., free_energy_distances, solver)
    end

    return merge(ws, (; Pref, W, A, A_init, Aadj, Aadj_init, Z, Zⁱ, expected_costs, free_energy_distances))
end

function _init_sparse(ws::NamedTuple, solver, cm, p::Problem, grid::Grid; verbose)
    gms = graph_measures(p)
    verbose && println("Initialising sparse factorizations...")
    Pref = _Pref(grid.affinities)
    W = _W(Pref, cm.θ, grid.costmatrix)
    # Sparse lhs
    A = I - W
    A_init = init(solver, A)
    Aadj, Aadj_init = if hastrait(needs_adjoint_init, gms)
        # Just take the adjoint of the factorization of A
        # where possible to save calculations and memory
        if hasproperty(A_init, :F)
            Aadj = A'
            # Use adjoint factorization of A rather than recalculating for A'
            Aadj_init = merge(A_init, (; F=A_init.F'))
            Aadj, Aadj_init
        else
            # LinearSolve.jl cant handle the adjoint 
            # so we duplicate work and allocations
            Aadj = sparse(A')
            Aadj_init = init(solver, Aadj)
            Aadj, Aadj_init
        end
    else
        nothing, nothing
    end

    CW = if hastrait(needs_expected_cost, gms) || connectivity_function(p) == ConScape.expected_cost
        grid.costmatrix .* W
    else
        nothing
    end

    return merge(ws, (; W, Pref, A, A_init, Aadj_init, Aadj, CW))
end

# All targets at once
_workspace_size(::MatrixSolver, g) = target_size(g)
# One target at a time
_workspace_size(::Union{VectorSolver,LinearSolver}, g) = first(target_size(g)), 1

isthreaded(s::Solver) = false
isthreaded(s::LinearSolver) = s.threaded
isthreaded(s::VectorSolver) = s.threaded

# Solver init
init(::Union{Nothing,MatrixSolver,VectorSolver}, A::AbstractMatrix) = (; F=lu(A))
function init(solver::VectorSolver, A::AbstractMatrix)
    F = lu(A)
    if isthreaded(solver)
        nbuffers = Threads.nthreads()
        channel = Channel{typeof(F)}(nbuffers)
        for _ in 1:nbuffers
            put!(channel, copy(F))
        end
        return channel
    else
        return F
    end
end
function init(solver::LinearSolver, A::AbstractMatrix)
    b = zeros(eltype(A), size(A, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, solver.args...; solver.keywords...)
    # TODO what is needed here?
    # Create a channel to store problem b vectors for threads
    # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    if isthreaded(solver)
        nbuffers = Threads.nthreads()
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
        return channel
    else
        return linsolve
    end
end

function LinearAlgebra.ldiv!(s::LinearSolver, init, B; B_copy)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    if isthreaded(s)
        channel = init
        # Get column memory from the channel
        linsolve = take!(channel)
        # Update solver with new b values
        reinit!(linsolve; b=vec(B_copy), reuse_precs=true)
        sol = LinearSolve.solve!(vec(B), linsolve, s.args...; s.keywords...)
        vec(B) .= sol.u
        put!(channel, linsolve)
    else
        linsolve = init
        reinit!(linsolve; b=vec(B_copy), reuse_precs=true)
        sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
        vec(B) .= sol.u
    end
    return B
end
LinearAlgebra.ldiv!(::Union{MatrixSolver,Nothing}, (; F), B; B_copy=copy(B)) =
    ldiv!(B, F, B_copy)
# LinearAlgebra.ldiv!(solver::Solver, A::AbstractMatrix, B::AbstractMatrix; kw...) = 
# ldiv!(solver, init(solver, A), B; kw...)
function LinearAlgebra.ldiv!(s::VectorSolver, init, B; B_copy)
    # for SparseArrays.UMFPACK._AqldivB_kernel!(Z, F, B, transposeoptype)
    transposeoptype = SparseArrays.LibSuiteSparse.UMFPACK_A

    # This is basically SparseArrays.UMFPACK._AqldivB_kernel!
    # But we unroll it to avoid copies or allocation of B
    if isthreaded(s)
        channel = init
        F = take!(channel)
        # Solve for the column
        SparseArrays.UMFPACK.solve!(vec(B), F, vec(B_copy), transposeoptype)
        # Reuse the workspace 
        put!(channel, F)
    else
        F = init
        SparseArrays.UMFPACK.solve!(vec(B), F, vec(B_copy), transposeoptype)
    end
    return B
end
# Utils

# We may have multiple distance_measures per
# graph_measure, but we want a single RasterStack.
# So we merge the names of the two layers

function _merge_to_stack(nt::NamedTuple{K}) where {K}
    unique_nts = map(K) do k
        _mergename(Val{k}(), nt[k])
    end
    # merge unique layers into a single RasterStack
    nt = merge(unique_nts...)
    if all(map(x -> x isa Raster, nt))
        return RasterStack(nt)
    else
        return nt # Cant return a RasterStack for these outputs 
    end
end

function _mergename(::Val{K1}, gm::NamedTuple{K2}) where {K1,K2}
    # Combine outer and inner names with an underscore
    joinedkeys = map(K2) do k2
        Symbol(K1, :_, k2)
    end
    # And rename the NamedTuple
    NamedTuple{joinedkeys}(map(_maybe_raster, values(gm)))
end
_mergename(::Val{K1}, gm) where {K1} =
# We keep the name as is
    NamedTuple{(K1,)}((_maybe_raster(gm),))

function _check_z(s, Z, W, g)
    # Check that values in Z are not too small:
    if hasproperty(s, :check) && s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end

# This duplicats some logic from gridrsp
function _proximities!(
    expected_costs::AbstractMatrix,
    cm::ConnectivityMeasure,
    p::Problem,
    grsp::GridRSP
)
    g = grsp.g
    dt = cm.distance_transformation
    if isnothing(dt)
        map!(inv(g.costfunction), proximities, expected_costs)
    else
        map!(dt, proximities, expected_costs)
    end
    maybe_set_diagonal!(proximities, diagvalue(p), g.targetnodes)
    return proximities
end

# This only makes sense if arrays are sorted large to small
function _reshape(A::Array, dims::Tuple{Vararg{Int}})
    len = prod(dims)
    if size(A) == dims
        A
    else # if length(A) >= len
        # TODO make sure this doesn't allocate when the array is larger
        # We may need julia 1.11 to do this properly
        v = vec(A)
        resize!(v, len)
        reshape(v, dims)
        # else
        # error("Arrays were not sorted. Current len: $(length(A)), needed len: $len")
    end
end
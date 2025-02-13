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
struct LinearSolver <: Solver 
    args
    keywords
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

# In `init!` we allocate all large dense arrays 
function init!(
    ws::NamedTuple, 
    solver::Solver, 
    cm::FundamentalMeasure, 
    p::AbstractProblem,
    rast::RasterStack;
    verbose=false,
) 
    verbose && println("Defining grid for RasterStack size $(size(rast))...")
    grid = g = Grid(p, rast)
    verbose && println("Retreiving measures...")
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
    # TODO handle mixed distance functions
    outputs = if cm.distance_transformation isa NamedTuple
        map(gms) do gm
            if needs_connectivity(gm)
                map(cm.distance_transformation) do dt
                    returntype(gm) isa ReturnsDenseSpatial ? fill(0.0, size(rast)) : nothing
                end
            else
                fill(0.0, size(rast))
            end
        end
    else
        map(gms) do gm
            returntype(gm) isa ReturnsDenseSpatial ? fill(0.0, size(rast)) : nothing
        end
    end
    
    verbose && println("Finished allocating...")

    return (; Z, Zⁱ, workspaces, permuted_workspaces, g=grid, grid, free_energy_distances, expected_costs, proximities, outputs)
end
function init!(
    workspace::NamedTuple, s::Solver, cm::ConnectivityMeasure, p::AbstractProblem, rast::RasterStack;
    verbose=false,
) 
    # TODO what is needed here?
    return (; grid=Grid(p, rast))
end

# Solver init
init(::Union{Nothing,Solver}, A::AbstractMatrix) = (; F=lu(A))
# function init(s::VectorSolver, A::AbstractMatrix)
#     F = lu(A)
#     Tb = Vector{eltype(A)}
#     # if isthreaded(s)
#     #     # Create one init per thread
#     #     # UMFPACK `copy` shares memory but avoids workspace race conditions
#     #     nbuffers = Threads.nthreads()
#     #     [
#     #         (; 
#     #             F=(i == 1 ? F : copy(F)), 
#     #             b=Tb(undef, size(A, 2))
#     #         )
#     #         for i in 1:nbuffers
#     #     ]
#     # else
#         b = Tb(undef, size(A, 2))
#         return (; F, b)
#     # end
# end
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

# RSP is not used for ConnectivityMeasure, so the solver isn't used
function solve!(
    workspace::NamedTuple, 
    s::Union{MatrixSolver,LinearSolver}, 
    cm::ConnectivityMeasure, 
    p::AbstractProblem;
    verbose=false
) 
    g = workspace.g
    return map(p.graph_measures, workspace.outputs) do gm, output
        compute(gm, p, ; workspace..., output)
    end
end
function solve!(
    ws::NamedTuple,
    solver::Union{MatrixSolver,LinearSolver}, 
    cm::FundamentalMeasure, 
    p::Problem;
    verbose=false,
) 
    ws1 = _init_sparse(ws, solver, cm, p, ws.grid; verbose)
    ws2 = _solve_dense!(ws1, solver, cm, p; verbose)
    gms = graph_measures(p)
    results = _solve!(ws2, solver, cm, cm.distance_transformation, gms, p; verbose)
    return _merge_to_stack(results)
end
function solve!(
    ws::NamedTuple,
    solver::VectorSolver, 
    cm, 
    p::Problem;
    verbose=false,
) 
    # Get grid and preallocated vectors
    (; g) = ws
    gms = graph_measures(p)
    # Predefine min-vectors for targets (not worth putting in the workspace)
    target_qualities = g.target_qualities[g.targetnodes[1]]
    targetidx = g.targetidx[1:1]
    targetnodes = g.targetnodes[1:1]
    qt = g.qt[1:1]
    target_allocs = (; target_qualities, targetidx, targetnodes, qt)
    _update_targets!(target_allocs, g, 1)
    target_properties = (; targetidx, targetnodes, qt)
    target_grid = ConstructionBase.setproperties(g, target_properties)
    first = true
    ws1 =_init_sparse(ws, solver, cm, p, target_grid; verbose)
    ws2 = merge(ws1, (; grid=target_grid, g=target_grid))
    target_ws = ConstructionBase.setproperties(ws2, (; g=target_grid, grid=target_grid))
    target_ws1 = _solve_dense!(target_ws, solver, cm, p; verbose)
    result1 = _solve!(target_ws1, solver, cm, cm.distance_transformation, gms, p; verbose)
    target_results = Vector{typeof(result1)}(undef, length(g.targetnodes))
    target_results[1] = result1
    # solve one target at a time
    for i in eachindex(g.targetnodes)[2:end]
        target_qualities = g.target_qualities[g.targetidx[i]]
        _update_targets!(target_allocs, g, i)
        first = false
        # And rebuild the workspace with the new grid
        target_ws = ConstructionBase.setproperties(ws2, (; g=target_grid, grid=target_grid))
        # Use the matrix solve on this smaller problem
        target_ws1 = _solve_dense!(target_ws, solver, cm, p; verbose)
        result = _solve!(target_ws1, solver, cm, cm.distance_transformation, gms, p; verbose)
        target_results[i] = result
    end
    return _merge_to_stack(_maybe_raster(ws.outputs, g))
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
    return _combine_nested_flat(gms, nested, flat)
end
Base.@assume_effects :foldable function _combine_nested_flat(
    gms::NamedTuple{GMS}, nested, flat
) where GMS
    # Combine nested and flat results
    map(GMS) do k
        f = flat[k]
        if isnothing(f) 
            map(n -> n[k], nested)
        else
            f
        end
    end |> NamedTuple{GMS}
end
function _solve!(workspace, solver, cm, dt, gms::NamedTuple{GMS}, p; verbose) where GMS
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
            Aadj_init = init(solver, Aadj)
            Aadj_init, Aadj
        end
        Aadj_init, Aadj
    else
        nothing, nothing
    end

    CW = grid.costmatrix .* W

    return merge(ws, (; W, Pref, A, A_init, Aadj_init, Aadj, CW))
end

_workspace_size(::Solver, g) = size(g.costmatrix, 1), length(g.targetnodes)
# Vector solver is one target at a time
_workspace_size(::VectorSolver, g) = size(g.costmatrix, 1), 1

isthreaded(s::Solver) = false
isthreaded(s::LinearSolver) = s.threaded
isthreaded(s::VectorSolver) = s.threaded

function LinearAlgebra.ldiv!(s::LinearSolver, (; linsolve, channel, b), B)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    if isthreaded(s)
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
LinearAlgebra.ldiv!(::Union{MatrixSolver,VectorSolver,Nothing}, (; F), B; B_copy=copy(B)) = 
    ldiv!(B, F, B_copy)
# LinearAlgebra.ldiv!(solver::Solver, A::AbstractMatrix, B::AbstractMatrix; kw...) = 
    # ldiv!(solver, init(solver, A), B; kw...)
# function LinearAlgebra.ldiv!(s::VectorSolver, init, B; B_copy=nothing)
#     # for SparseArrays.UMFPACK._AqldivB_kernel!(Z, F, B, transposeoptype)
#     transposeoptype = SparseArrays.LibSuiteSparse.UMFPACK_A

#     # This is basically SparseArrays.UMFPACK._AqldivB_kernel!
#     # But we unroll it to avoid copies or allocation of B
#     if isthreaded(s)
#         channel = Channel{typeof(init[1])}(length(init))
#         for x in init
#             put!(channel, x)
#         end
#         # Create a channel to store problem b vectors for threads
#         # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
#         Threads.@threads for col in 1:size(B, 2)
#             # Get a workspace from the channel
#             F_t, b_t = take!(channel)
#             # Copy a column from B
#             b_t .= view(B, :, col)
#             # Solve for the column
#             SparseArrays.UMFPACK.solve!(view(B, :, col), F_t, b_t, transposeoptype)
#             # Reuse the workspace 
#             put!(channel, (F_t, b_t))
#         end
#     else
#         (; F, b) = init[1]
#         for col in 1:size(B, 2)
#             b .= view(B, :, col)
#             SparseArrays.UMFPACK.solve!(view(B, :, col), F, b, transposeoptype)
#         end
#     end

#     return B
# end
# Utils

# We may have multiple distance_measures per
# graph_measure, but we want a single RasterStack.
# So we merge the names of the two layers

function _merge_to_stack(nt::NamedTuple{K}) where K
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

_maybe_raster(x) = x
_maybe_raster(x::Raster) = x
_maybe_raster(x::Number) = Raster(fill(x), ())
_maybe_raster(mat::Raster, g) = mat
_maybe_raster(mat::AbstractMatrix, g::Union{Grid,GridRSP}) = 
    _maybe_raster(mat, dims(g))
_maybe_raster(mats::NamedTuple, g::Union{Grid,GridRSP}) = 
    map(mat -> _maybe_raster(mat, g), mats)
_maybe_raster(mat::AbstractMatrix, ::Nothing) = mat
_maybe_raster(mat::AbstractMatrix, dims::Tuple) = Raster(mat, dims)

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
        map!(inv(g.costfunction), proximities, expected_costs)
    else
        map!(dt, proximities, expected_costs)
    end
    _maybe_set_diagonal!(proximities, g, diagvalue(p))
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
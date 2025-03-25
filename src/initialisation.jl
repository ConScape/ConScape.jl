"""
    MultiGridInit

Holds multiple grids for the same regions, splitting
them into subgraphs.

Also allocates and holds workspaces and outputs,
as `MultiGridInit` is the level at which
whole spatial problems are solved. 

`solve` on `MultiGridInit` iterates over 
subgraphs, generating a `GridInit` for each
and solving it into the same output object.
"""
struct MultiGridInit{P<:Problem,G<:Grid,SG<:Grid,W<:AbstractVector,S<:Dict,O} <: Initialisation
    problem::P
    grid::G
    subgrids::Vector{SG}
    workspaces::Workspaces{W}
    storage::S
    outputs::O
end
MultiGridInit(problem::Problem, rast::RasterStack; kw...) =
    MultiGridInit(problem, Grid(problem, rast); kw...)
function MultiGridInit(problem::Problem, grid::Grid; 
    workspaces=nothing, verbose=false
)
    if !isnothing(grain(problem))
        grid = coarse_graining(grid, grain(problem))
    end
    subgrids = split_subgraphs(grid)
    workspaces = _allocate_workspaces!(workspaces, problem, first(subgrids))
    storage = _newstoragedict(workspaces)
    # Create a MultiGridInit without outputs
    mgi = MultiGridInit(problem, grid, subgrids, workspaces, storage, nothing)
    # Generate outputs for each graph measure, the first subgraph is the largest
    outputs = allocate_output(problem, mgi)
    # Now create a MultiGridInit with outputs
    return MultiGridInit(problem, grid, subgrids, workspaces, storage, outputs)
end

problem(mgi::MultiGridInit) = mgi.problem
grid(mgi::MultiGridInit) = mgi.grid
subgrids(mgi::MultiGridInit) = mgi.subgrids
workspaces(mgi::MultiGridInit) = mgi.workspaces
outputs(mgi::MultiGridInit) = mgi.outputs
storage(mgi::MultiGridInit) = mgi.storage

"""
    GridInit

Precalculated variables and outputs buckets used 
in `solve` and `compute` methods.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables such as sparse factorization once for all targets.
"""
struct GridInit{MM,P<:Problem{MM},G<:Grid,O<:Union{NamedTuple,Tuple},W<:AbstractArray,S<:Dict,Pr} <: Initialisation
    problem::P
    grid::G
    outputs::O
    workspaces::Workspaces{W}
    storage::S
    precalculation::Pr
    function GridInit(
        problem::P, grid::G, outputs::O, workspaces::Workspaces{W}, storage::S, precalculation::Pr
    ) where {P<:Problem{MM},G,O,W,S,Pr} where MM
        @assert length(workspaces) == nsources(grid)
        free!(workspaces)
        empty!(storage)
        new{MM,P,G,O,W,S,Pr}(problem, grid, outputs, workspaces, storage, precalculation)
    end
end

function GridInit(problem::Problem, grid::Grid;
    outputs=allocate_output(problem, grid),
    workspaces=nothing, 
    storage=nothing,
    verbose=false,
)
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    if isnothing(storage) 
        storage = _newstoragedict(workspaces)
    end
    precalculation = gridinit_precalculation(problem, grid)
    return GridInit(problem, grid, outputs, workspaces, storage, precalculation)
end
function GridInit(mgi::MultiGridInit, subgrid_id::Int; kw...)
    GridInit(problem(mgi), subgrids(mgi)[subgrid_id]; 
        workspaces=workspaces(mgi),
        outputs=outputs(mgi), 
        storage=storage(mgi), 
    )
end

grid(gi::GridInit) = gi.grid
problem(gi::GridInit) = gi.problem
probability(gi::GridInit) = gi.probability
workspaces(gi::GridInit) = gi.workspaces
outputs(gi::GridInit) = gi.outputs
storage(gi::GridInit) = gi.storage

"""
    TargetInit

Abstract type for precalculated variables at the level of single targets.

Dense vector variables like `Z` (fundamental matrix) are generated 
on demand in `getproperty` (e.g. `tp.Z`) and stored for subsequent requests.

These variables use preallocated [`Workspaces`](@ref) to avoid allocations.

Variables from the parent `GridInit` can also be accessed with
`getpropery`, e.g. `tp.W` returns a sparse matrix calculated for all targets.

Stores outputs and lazily calculated variables for use in `RandomisedShortestPath`-based measures.

Varables can be accessed with `getproperty`: `rsp_tp.Z`.

From the parent `Grid`, the available variables are:
`qᵗ`, `qˢ`, `A`, `C`

From the parent `GridInit`, the available variables are:

`probability`, `W`, `IW`, `CW`, `IW_factorization`, `IW_adj`, `IW_adj_factorization`,

For target dense vectors:

`Z`, `Zⁱ`, `QZⁱ`, `K`, `M`, `MZⁱ`, `Zrows`,
`expected_costs`, `free_energy_distances`, `survival_probabilities`, `power_mean_proximities`,
"""
struct TargetInit{MM,GI<:GridInit{MM}} <: Initialisation
    gridinit::GI
    target::TargetID
    function TargetInit(gi::GI, target::TargetID) where GI<:GridInit{MM} where MM
        @assert (length(workspaces(gi)) == nsources(gi) == sparse_size(gi)[1]) 
        free!(workspaces(gi))
        empty!(storage(gi))
        new{MM,GI}(gi, target)
    end
end
TargetInit(gi::GridInit, target::Int) = TargetInit(gi, target_ids(gi)[target])

gridinit(tp::TargetInit) = getfield(tp, :gridinit)
target(tp::TargetInit) = getfield(tp, :target)

outputs(tp::TargetInit) = outputs(gridinit(tp))
grid(tp::TargetInit) = grid(gridinit(tp))
problem(tp::TargetInit) = problem(gridinit(tp))
storage(tp::TargetInit) = storage(gridinit(tp))
workspaces(tp::TargetInit) = workspaces(gridinit(tp))

proximity_measure(mm::TargetInit) = proximity_measure(problem(mm))
diagvalue(mm::TargetInit) = diagvalue(problem(mm))
approx(mm::TargetInit) = approx(problem(mm))
theta(mm::TargetInit) = theta(problem(mm))

# All TargetInit allow retreiving proberties with `getproperty`
# from the parent `GridInit` or calculated and stored in 
# the `TargetInit`
function Base.getproperty(tp::TargetInit, x::Symbol)
    if x === :workspace
        return take!(workspaces(tp))
    elseif x === :qᵗ
        return target_quality_vector(tp)[target(tp).id]
    elseif x === :qˢ
        return source_quality_vector(tp)
    elseif x === :A
        return affinitymatrix(tp)
    elseif x === :C
        return costmatrix(tp)
    elseif hasproperty(gridinit(tp).precalculation, x)
        return Base.getproperty(gridinit(tp).precalculation, x)
    end
    # Defer to `get_or_compute` for all other properties
    # We wrap the output in a `ReadOnlyArray` to prevent bugs.
    return get_or_compute(tp, x)
end

function gridinit_precalculation(problem::Problem{<:RSP}, grid::Grid)
    probability = _probabilitymatrix(affinitymatrix(grid))
    W = _W(probability, theta(problem), costmatrix(grid))
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    if solver(problem) isa VectorSolver
        IW_adj = IW'
        # Use adjoint factorization of A rather than recalculating for A'
        IW_adj_factorization = IW_factorization'
    else # LinearSolver
        # LinearSolve.jl cant handle the adjoint 
        # so we duplicate work and allocations
        IW_adj = sparse(IW')
        IW_adj_factorization = init(solver(problem), IW_adj)
    end
    CW = costmatrix(grid) .* W

    return (; probability, W, IW, CW, IW_factorization, IW_adj, IW_adj_factorization)
end
function gridinit_precalculation(::Problem{<:LeastCost}, grid::Grid)
    probability = _probabilitymatrix(affinitymatrix(grid))
    cost_weighted_digraph = SimpleWeightedDiGraph(costmatrix(grid))
    (; probability, cost_weighted_digraph)
end
function gridinit_precalculation(problem::Problem{<:RandomWalk}, grid::Grid)
    stationary_distribution = _stationary_distribution(grid.Pref, solver(problem))
    probability = _probabilitymatrix(affinitymatrix(grid))
    return (; probability, stationary_distribution)
end

@inline function get_or_compute(tp::TargetInit{<:RSP}, x::Symbol)::Vector{Float64}
    st = storage(tp)
    if haskey(st, x)
        return st[x]
    end
    output = if x === :Z # "fundamental matrix"
        _fundamentalmatrix(tp)
    elseif x === :Zⁱ # elementwise inverse of Z
        _inv!(tp.workspace, tp.Z)
    elseif x === :Q
        (; qˢ, qᵗ, workspace) = tp
        workspace .= qˢ .* qᵗ
    elseif x === :QZⁱ 
        (; Q, Zⁱ, workspace) = tp
        workspace .= Q .* Zⁱ
    elseif x === :K
        _proximitymatrix(tp)
    elseif x === :M
        (; qˢ, K, qᵗ, workspace) = tp
        workspace .= qˢ .* K .* qᵗ
    elseif x === :MZⁱ 
        (; M, Zⁱ, workspace) = tp
        workspace .= M .* Zⁱ 
    elseif x === :Zrows
        _fundamental_rows(tp)
    elseif x === :expected_costs
        compute(ExpectedCost(), tp)
    elseif x === :free_energy_distances
        compute(FreeEnergyDistance(), tp)
    elseif x === :survival_probabilities
        compute(SurvivalProbability(), tp)
    elseif x === :power_mean_proximities
        compute(PowerMeanProximity(), tp)
    else
        error("Unknown property $x")
    end
    st[x] = output
    return output
end
@inline function get_or_compute(tp::TargetInit{<:RandomWalk}, x::Symbol)::Vector{Float64}
    # Either retrieve from storage, or calculate and store
    get!(storage(tp), x) do
        if x === :Z
            inv(IP .+ p')
        elseif x === :H
            (diag(Z)' .- Z) ./ p'
        else
            error("Unknown property $x")
        end
    end
end
@inline function get_or_compute(tp::TargetInit{<:LeastCost}, x::Symbol)::Vector{Float64}
    # Either retrieve from storage, or calculate and store
    get!(storage(tp), x) do
        if x == :shortest_paths
            Graphs.dijkstra_shortest_paths(gridinit(tp).cost_weighted_digraph, target(tp).spatial)
        elseif x == :shortest_paths_en
            Graphs.enumerate_paths(tp.shorted_paths)
        elseif x == :K # "proximity matrix"
            (; shortest_paths, workspace) = tp
            if isnothing(distance_transformation(tp))
                workspace1 .= 1.0 # TODO is this right? not shortest_paths.dists?
            else
                workspace1 .= distance_transformation(cm).(shortest_paths.dists)
            end
        elseif x === :M # "landscape matrix"
            (; qˢ, K, qᵗ, workspace) = tp
            workspace .= qˢ .* K .* qᵗ
        else
            error("Unknown property $x")
        end
    end
end

# Variable generation for GridInit
function _probabilitymatrix(A::SparseMatrixCSC)
    source_sums = vec(sum(A, dims=2))
    source_scaling = inv.(source_sums)
    return Diagonal(source_scaling) * A
end
function _W(Pref::SparseMatrixCSC, θ::Real, C::SparseMatrixCSC)
    LinearAlgebra.checksquare(Pref)
    W = Pref .* exp.((-).(θ) .* C)
    replace!(W.nzval, NaN => 0.0)
    return W
end

# Custom `inv` broadcast that avoids Inf
_inv(Z::AbstractArray) = _inv!(similar(Z), Z)
function _inv!(Zⁱ::AbstractArray, Z::AbstractArray)
    broadcast!(Zⁱ, Z) do x
        x = inv(x)
        isfinite(x) ? x : floatmax(eltype(Z))
    end
end

# Variable generation for TargetInit
function _proximitymatrix(tp::TargetInit{<:RSP})
    pm = proximity_measure(tp)
    proximities = compute(pm, tp)
    if pm isa DistanceMeasure
        dt = distance_transformation(tp)
        if !isnothing(dt)
            proximities .= dt.(proximities)
        end
    end
    _maybe_set_diagonal!(proximities, diagvalue(tp), target(tp).node)
    return proximities
end
function _fundamentalmatrix(tp::TargetInit{<:RSP})
    workspace1, workspace2 = workspaces(tp)
    b = _rhs!(workspace1, nsources(tp), target(tp))
    b_copy = _rhs!(workspace2, nsources(tp), target(tp))
    return ldiv!(solver(tp), b, tp.IW_factorization, b_copy)
end
function _fundamental_rows(tp::TargetInit{<:RSP})
    (; IW_adj_factorization) = tp
    b, b_copy = workspaces(tp)
    _rhs!(b, nsources(tp), target(tp))
    _rhs!(b_copy, nsources(tp), target(tp))
    ldiv!(solver(tp), b, IW_adj_factorization, b_copy)
end
function _check_z(tp::TargetInit{<:RSP})
    # Check that values in Z are not too small
    # TODO: does this make sense for single targets
    if check(tp) && minimum(tp.Z) * minimum(nonzeros(tp.CW)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end

# Computes the stationary distribution of a random walk following the transition probability matrix
function _stationary_distribution(P::SparseMatrixCSC, solver::Solver)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI[1, :] .= 1
    v = zeros(n)
    v[1] = 1
    return ldiv!(solver, PI, v)
end

# `init` and `solve`

init(m::Union{Measure,Tuple,NamedTuple}, problem::Problem, rast::RasterStack; kw...) = 
    init(m, problem, init(problem, rast); kw...)
init(problem::Problem, rast::RasterStack; kw...) = MultiGridInit(problem, rast; kw...)
init(problem::Problem, grid::Grid; kw...) = MultiGridInit(problem, grid; kw...)
init(gi::GridInit, target::Union{Int,TargetID}) = TargetInit(gi, target)
init(mgi::MultiGridInit, subgrid_id::Int; kw...) = GridInit(mgi, subgrid_id; kw...)

# Allow solving all the levels of precalculated object with specific measures
solve(p::Initialisation; kw...) = solve(measures(p), p; kw...)
solve(measure::Measure, mgi::Initialisation; kw...) =
    only(values(solve((measure,), mgi; kw...)))
solve(measures::Union{NamedTuple,Tuple}, g::Grid; verbose=false, kw...) = 
    solve(Problem(measures; kw...), g::Grid; verbose)
solve(measure::Measure, movement_mode::MovementMode, g::Grid; verbose=false, kw...) = 
    only(solve(Problem((m=measure,); movement_mode, kw...), g::Grid; verbose))
function solve(measures::Union{NamedTuple,Tuple}, mgi::MultiGridInit; 
    outputs=_maybe_new_outputs(measures, mgi), kw...
)
    # Loop over unnconnected subgraphs (there may be only one)
    for subgrid_id in eachindex(subgrids(mgi))
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        solve(measures, init(mgi, subgrid_id; outputs))
    end
    out = _maybe_raster(outputs, mgi)
    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end
function solve(measures::Union{NamedTuple,Tuple}, gi::GridInit; 
    outputs=_maybe_new_outputs(measures, gi), kw...
)
    # Then loop over targets
    for target_id in target_ids(gi)
        # Precalculate for this target and graph measures
        solve(measures, init(gi, target_id))
    end
    return _maybe_raster(outputs, gi)
end
function solve(measures::Union{NamedTuple,Tuple}, ti::TargetInit;
    outputs=outputs(ti)
)
    # Compute everything for this target and graph measures
    map(measures, outputs) do measure, output
        # Compute a measure for this target
        v = compute(measure, ti)
        # Write values to output object
        update_output!(output, measure, v, ti)
    end
end

# General utility functions

_maybe_set_diagonal!(proximitymatrix, diagvalue::Nothing, targetnodes) = nothing
function _maybe_set_diagonal!(proximitymatrix, diagvalue::Number, targetnodes::AbstractVector)
    for (j, i) in enumerate(targetnodes)
        proximitymatrix[i, j] = diagvalue
    end
end
_maybe_set_diagonal!(proximitymatrix, diagvalue::Number, targetnode::Int) = 
    proximitymatrix[targetnode] = diagvalue

# Fill a vector with zeros, and one for the target node
function _rhs!(workspace, n::Int, target::TargetID)
    fill!(workspace, 0.0)
    workspace[target.node] = 1.0
    return workspace
end

# Reshape arrays to a new size dstructively
# This only makes sense if arrays are sorted large to small
function _reshape!(A::Array, size::Tuple{Vararg{Int}})
    len = prod(size)
    if Base.size(A) == size
        A
    else # if length(A) >= len
        # TODO make sure this doesn't allocate when the array is larger
        # We may need julia 1.11 to do this properly
        v = vec(A)
        resize!(v, len)
        reshape(v, size)
    end
end

_allocate_workspaces!(x, problem::Problem, grid::Grid) =
    _allocate_workspaces!(x, problem, nsources(grid))
_allocate_workspaces!(x::Nothing, problem::Problem, length::Int) =
    Workspaces(length, nworkspaces(problem) + 20)
_allocate_workspaces!(workspaces::Workspaces, ::Problem, length::Int) =
    free!(resize!(workspaces, length))

_maybe_new_outputs(mes, mgi) =
    mes === measures(mgi) ? outputs(mgi) : allocate_output(mes, mgi)

_newstoragedict(::Workspaces{W}) where W = Dict{Symbol,W}()
const TargetID = @NamedTuple{spatial::CartesianIndex{2},grid_id::Int,subgrid_id::Int,node::Int}
const SourceID = CartesianIndex{2}

abstract type Initialisation end

costfunction(p::Initialisation) = costfunction(grid(p))
costmatrix(p::Initialisation) = costmatrix(grid(p))
affinitymatrix(p::Initialisation) = affinitymatrix(grid(p))
source_quality_vector(p::Initialisation) = source_quality_vector(grid(p))
target_quality_vector(p::Initialisation) = target_quality_vector(grid(p))
source_quality_spatial(p::Initialisation) = source_quality_spatial(grid(p))
target_quality_spatial(p::Initialisation) = target_quality_spatial(grid(p))
source_ids(p::Initialisation) = source_ids(grid(p))
target_ids(p::Initialisation) = target_ids(grid(p))
movement_mode(p::Initialisation) = movement_mode(problem(p))
solver(p::Initialisation) = solver(problem(p))
measures(p::Initialisation) = measures(problem(p))
proximity_measure(p::Initialisation) = proximity_measure(problem(p))
distance_transformation(p::Initialisation) = distance_transformation(problem(p))
diagvalue(p::Initialisation) = diagvalue(problem(p))
approx(p::Initialisation) = approx(problem(p))
theta(p::Initialisation) = theta(problem(p))

nsources(p::Initialisation) = length(source_ids(p))
ntargets(p::Initialisation) = length(target_ids(p))
sparse_size(p::Initialisation) = nsources(p), ntargets(p) 

Base.size(p::Initialisation) = Base.size(grid(p))
DimensionalData.dims(p::Initialisation) = dims(grid(p))

"""
    Grid(size::Tuple{Int,Int}; kw...)

Construct a `Grid` from an `affinitymatrix` matrix of type `SparseMatrixCSC`. 

# Keywords

- `affinitymatrix`: nothing
- `qualities::Matrix`: ones(nrows, ncols)
- `source_qualities::Matrix`: qualities
- `target_qualities::AbstractMatrix`: qualities
- `costfunction`: `MinusLog()` by default.
- `costmatrix`: optionally specify a sparse cost matrix. 
    By default it is calculated from `costfunction.(affinitymatrix)`
- `prune`: if the affinity and cost matrices will be pruned 
    to exclude unreachable nodes. `true` by default.

It is possible to also supply matrices of `source_qualities` and `target_qualities` as well as

Alternatively, it is possible to supply a matrix to `costs` directly. If `prune=true` (the default), 
"""
struct Grid{D<:Union{Tuple,Nothing},F<:Union{Nothing,Transformation},SQ,TQ} <: Initialisation
    size::Tuple{Int,Int}
    costfunction::F
    costmatrix::SparseMatrixCSC{Float64,Int}
    affinitymatrix::SparseMatrixCSC{Float64,Int}
    source_quality_spatial::SQ
    target_quality_spatial::TQ
    source_quality_vector::Vector{Float64}
    target_quality_vector::Vector{Float64}
    source_ids::Vector{SourceID}
    target_ids::Vector{TargetID}
    dims::D
end
Grid(nrows::Int, ncols::Int; kw...) = Grid((nrows, ncols); kw...)
function Grid(size::Tuple{Int,Int};
    affinitymatrix::SparseMatrixCSC{Float64,Int},
    qualities::AbstractMatrix=ones(size),
    source_qualities::AbstractMatrix=qualities,
    target_qualities::AbstractMatrix=qualities,
    costfunction::Union{Transformation,Nothing}=MinusLog(),
    costmatrix=mapnz(costfunction, affinitymatrix),
    check=false,
    prune=true,
)
    if prod(size) != LinearAlgebra.checksquare(affinitymatrix)
        n = Base.size(affinitymatrix, 1)
        throw(ArgumentError("grid size $size is incompatible with size of affinity matrix ($n, $n)"))
    end
    if prod(size) != LinearAlgebra.checksquare(costmatrix)
        n = Base.size(costmatrix, 1)
        throw(ArgumentError("grid size $size is incompatible with size of cost matrix ($n, $n)"))
    end

    # This is too expensive to calculate for small target grids
    if check
        if any(t -> t < 0, nonzeros(costmatrix))
            throw(ArgumentError("The cost graph can have only non-negative edge weights. Perhaps you should change the cost function?"))
        end
        cost_digraph = SimpleDiGraph(costmatrix)
        affinity_digraph = SimpleDiGraph(affinitymatrix)

        if ne(difference(cost_digraph, affinity_digraph)) > 0
            throw(ArgumentError("cost graph contains edges not present in the affinity graph"))
        end
    end

    source_quality_spatial = _prepare_qualities(source_qualities)
    target_quality_spatial = _prepare_qualities(target_qualities)

    # Initially just every node

    # Prune
    source_ids = vec(collect(CartesianIndices(size)))
    if prune
        nonzerocells = findall(!isnan ∘ !iszero, vec(sum(affinitymatrix, dims=1)))
        source_ids = source_ids[nonzerocells]
        affinitymatrix = affinitymatrix[nonzerocells, nonzerocells]
    end

    # Subset of source_ids with valid quality
    target_ids = _target_ids(target_qualities, source_ids)
    # Initially just all spatial source qualities
    source_quality_vector = vec(source_quality_spatial)
    # Subset of spatial target qualities with valid quality
    target_quality_vector = [target_quality_spatial[t.spatial] for t in target_ids]

    return Grid(
        size,
        costfunction,
        costmatrix,
        affinitymatrix,
        source_quality_spatial, target_quality_spatial,
        source_quality_vector, target_quality_vector,
        source_ids, target_ids,
        dims(source_qualities),
    )
end
function Grid(rast::RasterStack;
    affinitymatrix=ConScape.graph_matrix_from_raster(rast.affinities),
    source_qualities=rast.source_qualities,
    target_qualities=get(rast, :target_qualities, source_qualities),
    kw...
)
    Grid(size(rast); affinitymatrix, source_qualities, target_qualities, kw...)
end
Grid(p::AbstractProblem, rast::RasterStack; kw...) =
    Grid(rast; costfunction=costfunction(p), kw...)

affinitymatrix(g::Grid) = g.affinitymatrix
costmatrix(g::Grid) = g.costmatrix
costfunction(g::Grid) = g.costfunction
source_quality_spatial(g::Grid) = g.source_quality_spatial
target_quality_spatial(g::Grid) = g.target_quality_spatial
source_quality_vector(g::Grid) = g.source_quality_vector
target_quality_vector(g::Grid) = g.target_quality_vector
source_ids(g::Grid) = g.source_ids
target_ids(g::Grid) = g.target_ids

Base.size(g::Grid) = g.size
function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    print(io, summary(g), " of size ", g.size)
end

DimensionalData.dims(g::Grid) = g.dims


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
    workspaces = if length(subgrids) > 0
        _allocate_workspaces!(workspaces, problem, first(subgrids))
    else
        Workspaces(0, 0)
    end
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
function GridInit(mgi::MultiGridInit, subgrid_id::Int; 
    workspaces=workspaces(mgi),
    outputs=outputs(mgi), 
    storage=storage(mgi), 
    kw...
)
    GridInit(problem(mgi), subgrids(mgi)[subgrid_id]; 
        workspaces, outputs, storage, kw...
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

gridinit(ti::TargetInit) = getfield(ti, :gridinit)
target(ti::TargetInit) = getfield(ti, :target)

outputs(ti::TargetInit) = outputs(gridinit(ti))
grid(ti::TargetInit) = grid(gridinit(ti))
problem(ti::TargetInit) = problem(gridinit(ti))
storage(ti::TargetInit) = storage(gridinit(ti))
workspaces(ti::TargetInit) = workspaces(gridinit(ti))

proximity_measure(mm::TargetInit) = proximity_measure(problem(mm))
diagvalue(mm::TargetInit) = diagvalue(problem(mm))
approx(mm::TargetInit) = approx(problem(mm))
theta(mm::TargetInit) = theta(problem(mm))

# All TargetInit allow retreiving proberties with `getproperty`
# from the parent `GridInit` or calculated and stored in 
# the `TargetInit`
function Base.getproperty(ti::TargetInit, x::Symbol)
    if x === :workspace
        return take!(workspaces(ti))
    elseif x === :θ 
        return theta(ti)
    elseif x === :qᵗ
        return target_quality_vector(ti)[target(ti).subgrid_id]
    elseif x === :qˢ
        return source_quality_vector(ti)
    elseif x === :A
        return affinitymatrix(ti)
    elseif x === :C
        return costmatrix(ti)
    elseif hasproperty(gridinit(ti).precalculation, x)
        return Base.getproperty(gridinit(ti).precalculation, x)
    end
    # Defer to `get_or_compute` for all other properties
    # We wrap the output in a `ReadOnlyArray` to prevent bugs.
    return get_or_compute!(ti, x)
end

function gridinit_precalculation(problem::Problem{<:RSP}, grid::Grid)
    probability, A_rowsums = _probabilitymatrix(affinitymatrix(grid))
    W = _W(probability, theta(problem), costmatrix(grid))
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    Aⁱ = mapnz(inv, affinitymatrix(grid))
    # TODO: is cost and affinity the right way around here?
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

    return (; probability, W, IW, IW_adj, CW, IW_factorization, IW_adj_factorization, Aⁱ, A_rowsums)
end
function gridinit_precalculation(::Problem{<:LeastCost}, grid::Grid)
    probability, A_rowsums = _probabilitymatrix(affinitymatrix(grid))
    # TODO: use a raster based shortest path algorithm from Geomorphometry.jl
    # disjkstra is especially slow due to allocations,
    # searchsorted for index lookups, and Dict getindex/setindex!.
    cost_weighted_digraph = SimpleWeightedDiGraph(costmatrix(grid))
    dsp1 = dijkstra_shortest_paths(cost_weighted_digraph, 1)
    parents = dsp1.parents
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in 1:length(parents)]
    (; probability, A_rowsums, cost_weighted_digraph, path_allocs)
end
function gridinit_precalculation(problem::Problem{<:RandomWalk}, grid::Grid)
    probability, A_rowsums = _probabilitymatrix(affinitymatrix(grid))
    stationary_distribution = _stationary_distribution(solver(problem), probability)
    PC = probability .* costmatrix(grid)
    PC_rowsums = sum(PC; dims=2)
    IP = I - probability
    return (; probability, A_rowsums, stationary_distribution, PC, PC_rowsums, IP)
end

# Variable generation for GridInit
function _probabilitymatrix(A::SparseMatrixCSC)
    source_sums = vec(sum(A, dims=2))
    source_scaling = inv.(source_sums)
    P = Diagonal(source_scaling) * A
    return P, source_sums
end
function _W(Pref::SparseMatrixCSC, θ::Real, C::SparseMatrixCSC)
    LinearAlgebra.checksquare(Pref)
    W = Pref .* exp.(-θ .* C)
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
function _proximitymatrix(ti::TargetInit)
    pm = proximity_measure(ti)
    proximities = get_or_compute!(ti, pm)
    if pm isa DistanceMeasure
        dt = distance_transformation(ti)
        if !isnothing(dt)
            proximities .= dt.(proximities)
        end
    end
    _maybe_set_diagonal!(proximities, diagvalue(ti), target(ti).node)
    return proximities
end
function _fundamentalmatrix(ti::TargetInit{<:RSP})
    workspace1, workspace2 = workspaces(ti)
    b = _rhs!(workspace1, nsources(ti), target(ti))
    b_copy = _rhs!(workspace2, nsources(ti), target(ti))
    return ldiv!(solver(ti), b, ti.IW_factorization, b_copy)
end
function _fundamental_rows(ti::TargetInit{<:RSP})
    b = ti.workspace
    _rhs!(b, nsources(ti), target(ti))
    ldiv!(ti, ti.IW_adj_factorization, b)
end
function _check_z(ti::TargetInit{<:RSP})
    # Check that values in Z are not too small
    # TODO: does this make sense for single targets
    if check(ti) && minimum(ti.Z) * minimum(nonzeros(ti.CW)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end

# Computes the stationary distribution of a random walk following the transition probability matrix
function _stationary_distribution(solver::Solver, P::SparseMatrixCSC)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI_factorization = init(solver, PI)
    PI[1, :] .= 1
    v = zeros(n)
    v1 = zeros(n)
    v[1] = v1[1] = 1
    return ldiv!(solver, v, PI_factorization, v1)
end

# `init` and `solve`

init(movement_mode::MovementMode, grid::Grid; kw...) =
    init(Problem(; movement_mode), grid; kw...)
init(m::Union{Measure,Tuple,NamedTuple}, problem::Problem, rast::RasterStack; kw...) = 
    init(m, problem, init(problem, rast); kw...)
init(problem::Problem, rast::RasterStack; kw...) = MultiGridInit(problem, rast; kw...)
init(problem::Problem, grid::Grid; kw...) = MultiGridInit(problem, grid; kw...)
init(gi::GridInit, target::Union{Int,TargetID}) = TargetInit(gi, target)
init(mgi::MultiGridInit, subgrid_id::Int; kw...) = GridInit(mgi, subgrid_id; kw...)

solve(p::Problem, input::Union{Grid,RasterStack}; kw...) = 
    solve(init(p, input; kw...))
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
    out = map(outputs, measures) do o, m
        returntrait(m) isa DenseSpatial ? _maybe_raster(o, mgi) : o
    end
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
    return map(outputs, measures) do o, m
        returntrait(m) isa DenseSpatial ? _maybe_raster(o, gi) : o
    end
end
function solve(measures::Union{NamedTuple,Tuple}, ti::TargetInit;
    outputs=outputs(ti)
)
    # Compute everything for this target and graph measures
    map(measures, outputs) do measure, output
        # Dont compute the same measure multiple times
        v = get_or_compute!(ti, measure)
        # Write values to output object
        update_output!(output, measure, v, ti)
    end
end
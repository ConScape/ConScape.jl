"""
    Initialisation

Supertype for the initialisateion / solve sequence in Conscape.jl.

Initialisation is deeply nested in ConScape for two specific reasons.

1. A landscape can be made up of multiple disconnected subgraphs, but we
  need outputs that are the composite of all of them, e.g. a spatial map.
2. Within each connected graph the memory efficient approach to computing measure
  is to calculate one single target at at time, rather than using large matrix solves.

We attempt to allocate as much memory as possible for the whold grid, and
at each lower level

A further subtlety is that [`Level`](@ref) markers control what level of output is
allocated. In complete `solve!` runs this is usually [`GridGraphLevel`](@ref). Nut when a
user explicitly solves a conected subgraph or single target, [`ConnectedGraphLevel`](@ref)
or [`TargetLevel`](@ref) will be used so that the output is not expanded to the full
grid size when not needed.

[ 0. User input: [`ConScapeProblem`](@ref) and `RasterStack`/[`GridGraph`](@ref) ]
    │
    │   Components:
    │   • `ConScapeProblem`: defines the model parameters (movement modes, resistance, etc.).
    │   • `GridGraph` / `RasterStack`: provides the landscape data (habitat quality, permeability).
    │
    └─ `init(problem, data)` is called to combine the problem definition with the landscape data.
    ↓

[ 1. [`GridGraphInit`](@ref): the complete landscape initialised ]
    │
    │   Role: Represents the entire input landscape and assembles the complete output.
    │
    │   Allocations:
    │   • The final, full-sized output arrays for all measures.
    │   • A vector of `ConnectedGraph` objects, one for each "island".
    │   • Workspace arrays for all target computations (these may be resized later as needed)
    │
    │   Precalculation:
    │   • Identifies disconnected subgraphs ("islands") to iterate over during solve.
    │
    │   Output:
    │   • Stitches together results from all `ConnectedGraphInit` subgraphs
    │     into the final, complete outputs.
    │
    └─ For each "island" it initialises a `ConnectedGraphInit` and runs `solve!` for it.
    ↓

[ 2. [`ConnectedGraphInit`](@ref): an independent connected graph "Island" ]
    │
    │   Role: Manages measure compututations for a single, fully connected landscape graph.
    │
    │   Allocations:
    │   • The sparse matrices for the connected subgraph.
    │   • Matrix factorization objects (e.g., `LU`).
    │   • Subgraph level output arrays.
    │   • Intermediate storage for measures that aggregate results across all targets.
    │
    │   Precalculation:
    │   • Builds the sparse graph and performs expensive one-time matrix
    │     operations (e.g., LU factorization) for the specific *movement model*.
    │   • This step is done once per island and is **reused by all measures**.
    │
    │   Output:
    │   • Collects results from each `TargetInit` and transfers them back to the
    │     correct location in the `GridGraphInit`'s output rasters.
    │
    └─ For each target cell, it initialises a `TargetInit` and runs `solve!` for it.
    ↓

[ 3. [`TargetInit`](@ref): a single location ]
    │
    │   Role: Manages all computation for one target cell.
    │
    │   Allocations:
    │   • None.
    │
    │   Precalculation:
    │   • Manages a cache for reusable computation results via `get_or_compute_target!`.
    │
    │   Output:
    │   • Computes the final numerical result for each requested measure.
"""
abstract type Initialisation end

const TargetID = @NamedTuple{spatialidx::CartesianIndex{2},gridgraphidx::Int,connectedgraphidx::Int,node::Int}
const SourceID = CartesianIndex{2}

# Generic methods that forward to the ConScapeProblem
movement(i::Initialisation) = movement(problem(i))
costfunction(i::Initialisation) = costfunction(problem(i))
likelihoodfunction(i::Initialisation) = likelihoodfunction(problem(i))
solver(i::Initialisation) = solver(problem(i))
measures(i::Initialisation) = map(mol -> mol.measure, measures_outputs(i))
outputs(i::Initialisation) = map(mol -> mol.output, measures_outputs(i))
proximity_measure(i::Initialisation) = proximity_measure(problem(i))
distance_transformation(i::Initialisation) = distance_transformation(problem(i))
diagvalue(i::Initialisation) = diagvalue(problem(i))

# Getters that forward to the MovementMode
approx(i::Initialisation) = approx(movement(i))
theta(i::Initialisation) = theta(movement(i))

# Getters that forward to the GridGraph
gridgraph_size(i::Initialisation) = gridgraph_size(gridgraph(i))
target_size(i::Initialisation) = (ntargets(gridgraph(i)),)

# Base methods
Base.size(i::Initialisation, args...) = Base.size(gridgraph(i), args...)
Base.length(i::Initialisation) = Base.length(gridgraph(i))

# DimensionalData methods
DimensionalData.dims(i::Initialisation) = dims(gridgraph(i))

"""
    GridGraphInit

Holds multiple grids for the region, splitting it
them into connected subgraphs.

Also allocates and holds workspaces and outputs,
as `GridGraphInit` is the level at which
whole spatial problems are solved.

`solve` on `GridGraphInit` iterates over
connected subgraphs, generating a `ConnectedGraphInit` for each
and solving it into the same output object.

## Example

To construct a `GridGraphInit` :

```julia
ggi = init(problem, rast)
```

To solve measures in the `ConScapeProblem`:

```julia
results = solve(ggi)
```

To solve arbitrary measures:

```julia
ec = solve(ExpectedCost(), ggi)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), ggi)
```
"""
struct GridGraphInit{
    P<:ConScapeProblem,G<:GridGraph,CG<:ConnectedGraph,S<:Dict,O
} <: Initialisation
    problem::P
    gridgraph::G
    connectedgraphs::Vector{CG}
    workspaces::Workspaces{Vector{Float64}}
    mworkspaces::Workspaces{Matrix{Float64}}
    storage::S
    outputs::O
end
function GridGraphInit(problem::ConScapeProblem, rast::RasterStack; 
    workspaces=nothing,
    mworkspaces=nothing,
    finallevel=GridGraphLevel(),
    kw...
)
    return GridGraphInit(problem, GridGraph(problem, rast; kw...); 
        workspaces, mworkspaces, finallevel
    )
end
function GridGraphInit(problem::ConScapeProblem, gridgraph::GridGraph;
    workspaces=nothing,
    mworkspaces=nothing,
    finallevel=GridGraphLevel(),
)
    _check_inputs(problem, gridgraph)
    connectedgraphs = split_connected_graphs(gridgraph;
        costfunction=costfunction(problem),
        likelihoodfunction=likelihoodfunction(problem),
    )
    workspaces = if length(connectedgraphs) > 0
        _allocate_workspaces!(workspaces, problem, first(connectedgraphs))
    else
        Workspaces(0, 0)
    end
    mworkspaces = if length(connectedgraphs) > 0
        _allocate_mworkspaces!(mworkspaces, problem, first(connectedgraphs))
    else
        Workspaces(0, 0)
    end
    storage = _connectedgraph_storage(movement(problem), workspaces)
    # Only allocate outputs if requested at the grid graph level (all subgraphs)
    # Otherwise ConnectedGraphInit will do this further down.
    outputs = if finallevel isa GridGraphLevel
        allocate_gridgraph_output(
            measures(problem), gridgraph, connectedgraphs
        )
    end
    # Now create a GridGraphInit with outputs
    return GridGraphInit(
        problem, gridgraph, connectedgraphs, workspaces, mworkspaces, storage, outputs
    )
end

# Field getters
problem(ggi::GridGraphInit) = ggi.problem
gridgraph(ggi::GridGraphInit) = ggi.gridgraph
connectedgraphs(ggi::GridGraphInit) = ggi.connectedgraphs
workspaces(ggi::GridGraphInit) = ggi.workspaces
mworkspaces(ggi::GridGraphInit) = ggi.mworkspaces
storage(ggi::GridGraphInit) = ggi.storage
measures_outputs(ggi::GridGraphInit) = ggi.outputs

# Methods that forward to the ConnectedGraph
nconnectedgraphs(ggi::GridGraphInit) = length(connectedgraphs(ggi))

# Methods that forward to the GridGraph
stepcost(p::GridGraphInit) = stepcost(gridgraph(p))
steplikelihood(p::GridGraphInit) = steplikelihood(gridgraph(p))
sourcequality(p::GridGraphInit) = sourcequality(gridgraph(p))
targetquality(p::GridGraphInit) = targetquality(gridgraph(p))

# Sanity checks
_check_inputs(x, init) = _check_inputs(movement(x), init)
function _check_inputs(::RSP, g)
    isnothing(stepcost(g)) && throw(ArgumentError("GridGraph has no stepcost for RSP"))
    isnothing(steplikelihood(g)) && throw(ArgumentError("GridGraph has no steplikelihood for RSP"))
end
function _check_inputs(::LCP, g)
    isnothing(stepcost(g)) && throw(ArgumentError("GridGraph has no stepcost for LCP"))
end
function _check_inputs(::RandomWalk, g)
    isnothing(steplikelihood(g)) && throw(ArgumentError("GridGraph has no steplikelihood for LCP"))
end
_check_inputs(::Euclidean, g) = nothing

"""
    ConnectedGraphInit

Precalculated variables and outputs buckets used
in `solve` and `compute` methods.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables such as sparse factorization once for all targets.

## Example

To construct a `ConnectedGraphInit` :

```julia
probleminit = init(problem, rast)
si = init(probleminit, 1)
````

To solve measures in the `ConScapeProblem` for this connectedgraph:

```julia
results = solve(si)
```

To solve arbitrary measures for this connectedgraph:

```julia
ec = solve(ExpectedCost(), si)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), si)
```
"""
struct ConnectedGraphInit{
    MM,
    P<:ConScapeProblem{MM},
    GG<:GridGraph,
    CG<:ConnectedGraph,
    O<:Union{Nothing,NamedTuple,Tuple},
    S<:Dict,
    Pr
} <: Initialisation
    problem::P
    gridgraph::GG
    connectedgraph::CG
    outputs::O
    workspaces::Workspaces{Vector{Float64}}
    mworkspaces::Workspaces{Matrix{Float64}}
    storage::S
    precalculation::Pr
    connectedgraphid::Int
    # Internal constructor enforces:
    # 1. workspace length matches number of sources
    # 2. storage Dict is empty
    # 3. workspaces are freed for use
    function ConnectedGraphInit(
        problem::P,
        gridgraph::GG,
        connectedgraph::CG,
        outputs::O,
        workspaces::Workspaces{Vector{Float64}},
        mworkspaces::Workspaces{Matrix{Float64}},
        storage::S,
        precalculation::Pr,
        id::Int,
    ) where {P<:ConScapeProblem{MM},GG,CG,O,S,Pr} where MM
        @assert length(workspaces) == nsources(connectedgraph)
        empty!(storage)
        free!(workspaces)
        new{MM,P,GG,CG,O,S,Pr}(
            problem, gridgraph, connectedgraph, outputs, workspaces, mworkspaces, storage, precalculation, id
        )
    end
end
function ConnectedGraphInit(ggi::GridGraphInit, connectedgraphid::Int;
    finallevel=ConnectedGraphLevel(),
    workspaces=workspaces(ggi),
    mworkspaces=mworkspaces(ggi),
    storage=storage(ggi),
    kw...
)
    connectedgraph = connectedgraphs(ggi)[connectedgraphid]
    _check_inputs(ggi, connectedgraph)
    workspaces = _allocate_workspaces!(workspaces, problem(ggi), connectedgraph)
    mworkspaces = _allocate_mworkspaces!(mworkspaces, problem(ggi), connectedgraph)
    if isnothing(storage)
        storage = _connectedgraph_storage(movement(ggi), workspaces)
    end
    sparse_precalc = sparse_precalculation(problem(ggi), connectedgraph)
    outputs = allocate_connectedgraph_output(
        finallevel, measures(ggi), gridgraph(ggi), connectedgraph, sparse_precalc
    )

    # Partially initialise so we can use this in dense precalculation
    connectedgraphinit = ConnectedGraphInit(
        problem(ggi),
        gridgraph(ggi),
        connectedgraph,
        outputs,
        workspaces,
        mworkspaces,
        storage,
        sparse_precalc,
        connectedgraphid,
    )

    # Precalculate dense matrices where needed
    dense_precalc = dense_precalculation(connectedgraphinit)
    precalculation = merge(sparse_precalc, dense_precalc)

    return ConnectedGraphInit(
        problem(ggi),
        gridgraph(ggi),
        connectedgraph,
        outputs,
        workspaces,
        mworkspaces,
        storage,
        precalculation,
        connectedgraphid,
    )
end

# Field getters
gridgraph(sgi::ConnectedGraphInit) = getfield(sgi, :gridgraph)
connectedgraph(sgi::ConnectedGraphInit) = getfield(sgi, :connectedgraph)
problem(sgi::ConnectedGraphInit) = getfield(sgi, :problem)
workspaces(sgi::ConnectedGraphInit) = getfield(sgi, :workspaces)
mworkspaces(sgi::ConnectedGraphInit) = getfield(sgi, :mworkspaces)
measures_outputs(sgi::ConnectedGraphInit) = getfield(sgi, :outputs)
storage(sgi::ConnectedGraphInit) = getfield(sgi, :storage)
precalculation(sgi::ConnectedGraphInit) = getfield(sgi, :precalculation)
connectedgraphid(sgi::ConnectedGraphInit) = getfield(sgi, :connectedgraphid)

# Getters that forward to the ConnectedGraph
nsources(sgi::ConnectedGraphInit) = nsources(connectedgraph(sgi))
ntargets(sgi::ConnectedGraphInit) = ntargets(connectedgraph(sgi))
stepcost(sgi::ConnectedGraphInit) = stepcost(connectedgraph(sgi))
steplikelihood(sgi::ConnectedGraphInit) = steplikelihood(connectedgraph(sgi))
sourcequality(sgi::ConnectedGraphInit) = sourcequality(connectedgraph(sgi))
targetquality(sgi::ConnectedGraphInit) = targetquality(connectedgraph(sgi))
sourceids(sgi::ConnectedGraphInit) = sourceids(connectedgraph(sgi))
targetids(sgi::ConnectedGraphInit) = targetids(connectedgraph(sgi))
connectedgraph_size(sgi::ConnectedGraphInit) = connectedgraph_size(connectedgraph(sgi))

_connectedgraph_storage(::MovementMode, ::Workspaces{W}) where W<:AbstractArray{T} where T =
    Dict{Symbol,ReadOnlyArray{T,1,W}}()
# Need to store the Woodbury matrix
# TODO: find a better way to do this. Storage should be strongly typed
_connectedgraph_storage(::RandomWalk, ::Workspaces{W}) where W<:AbstractArray{T} where T =
    Dict{Symbol,Any}()

# Base.getproperty/propertynames let us use the `cgi.somevariable` 
# syntax in `finalize_connectedgraph_output` and similar methods
Base.@assume_effects :foldable @inline Base.getproperty(cgi::ConnectedGraphInit, x::Symbol) = 
    getproperty(precalculation(cgi), x)
Base.@assume_effects :foldable @inline Base.propertynames(cgi::ConnectedGraphInit) = 
    propertynames(precalculation(cgi))

"""
    TargetInit

Abstract type for precalculated variables at the level of single targets.

Dense vector variables like `Z` (fundamental matrix) are generated
on demand in `getproperty` (e.g. `tp.Z`) and stored for subsequent requests.

These variables use preallocated [`Workspaces`](@ref) to avoid allocations.

Variables from the parent `ConnectedGraphInit` can also be accessed with
`getproperty`, e.g. `tp.W` returns a sparse matrix calculated for all targets.

Stores outputs and lazily calculated variables for use in `RandomisedShortestPath`-based measures.

Varables can be accessed with `getproperty`: `rsp_tp.Z`.

From the parent `ConnectedGraph`, the available variables are:
`qᵗ`, `qˢ`, `L`, `C`

From the parent `ConnectedGraphInit`, the available variables are:

`P`, `W`, `IW`, `CW`, `IW_factorization`, `IW_adj`, `IW_adj_factorization`,

For target dense vectors:

`Z`, `Zⁱ`, `Zrows`, `Q`, `K`, `M`.

## Example

To construct a `TargetInit` :

```julia
probleminit = init(problem, rast)
connectedgraph = 1
connectedgraphinit = init(probleminit, connectedgraph)
target_idx = 7
ti = init(connectedgraphinit, target_idx)
````

To solve measures in the `ConScapeProblem` for this target:

```julia
results = solve(ti)
```

To solve arbitrary measures for this target:

```julia
ec = solve(ExpectedCost(), ti)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), ti)
```
"""
struct TargetInit{MovMode,CGI<:ConnectedGraphInit{MovMode}} <: Initialisation
    connectedgraphinit::CGI
    target::TargetID
    function TargetInit{MovMode,CGI}(cgi, target) where {MovMode,CGI}
        new{MovMode,CGI}(cgi, target)
    end
end
# Constructor enforces:
# 1. workspace length matches number of sources and connected graph size
# 2. storage Dict is empty
# 3. workspaces are freed for use
function TargetInit(
    cgi::CGI, target::TargetID
) where {CGI<:ConnectedGraphInit{MM}} where MM
    @assert (
        length(workspaces(cgi)) ==
        nsources(connectedgraph(cgi)) ==
        connectedgraph_size(cgi)[1]
    )
    free!(workspaces(cgi))
    empty!(storage(cgi))
    TargetInit{MM,CGI}(cgi, target)
end
function TargetInit(cgi::ConnectedGraphInit, target::CartesianIndex; kw...)
    i = findfirst(t -> t.spatialidx == target, targetids(cgi))
    isnothing(i) && throw(ArgumentError("target indices $target are not part of this network"))
    TargetInit(cgi, i; kw...)
end
TargetInit(cgi::ConnectedGraphInit, target::Int; kw...) =
    TargetInit(cgi, targetids(cgi)[target]; kw...)

# Field getter functions
connectedgraphinit(ti::TargetInit) = getfield(ti, :connectedgraphinit)
target(ti::TargetInit) = getfield(ti, :target)

# Getter functions that forward to the target id
targetnode(ti::TargetInit) = target(ti).node
targetspatialidx(ti) = target(ti).spatialidx
targetconnectedgraphidx(ti) = target(ti).connectedgraphidx

# Getter functions that forward to the ConnectedGraphInit
measures_outputs(ti::TargetInit) = measures_outputs(connectedgraphinit(ti))
gridgraph(ti::TargetInit) = gridgraph(connectedgraphinit(ti))
connectedgraph(ti::TargetInit) = connectedgraph(connectedgraphinit(ti))
problem(ti::TargetInit) = problem(connectedgraphinit(ti))
storage(ti::TargetInit) = storage(connectedgraphinit(ti))
workspaces(ti::TargetInit) = workspaces(connectedgraphinit(ti))
mworkspaces(ti::TargetInit) = mworkspaces(connectedgraphinit(ti))

# Getter functions that forward to the ConnectedGraph
nsources(sgi::TargetInit) = nsources(connectedgraph(sgi))
ntargets(sgi::TargetInit) = ntargets(connectedgraph(sgi))
stepcost(ti::TargetInit) = stepcost(connectedgraph(ti))
steplikelihood(ti::TargetInit) = steplikelihood(connectedgraph(ti))
sourcequality(ti::TargetInit) = sourcequality(connectedgraph(ti))
targetquality(ti::TargetInit) = targetquality(connectedgraph(ti))
sourceids(ti::TargetInit) = sourceids(connectedgraph(ti))
targetids(ti::TargetInit) = targetids(connectedgraph(ti))
connectedgraph_size(ti::TargetInit) = connectedgraph_size(connectedgraphinit(ti))
precalculation(ti::TargetInit) = precalculation(connectedgraphinit(ti))

# Workspace taker
workspace(ti::Initialisation) = take!(workspaces(ti))
mworkspace(ti::Initialisation) = take!(mworkspaces(ti))

# Reuse this DimensionalData method
# TODO: reolve/combine with setmeasures
function Rasters.rebuild(ti::TargetInit;
    connectedgraphinit=connectedgraphinit(ti),
    target=target(ti),
)
    _rebuild(ti, connectedgraphinit, target)
end

@noinline function _rebuild(
    ti::TargetInit, connectedgraphinit::CGI, target::TargetID
) where {CGI<:ConnectedGraphInit{MM}} where {MM}
    TargetInit{MM,CGI}(connectedgraphinit, target)
end

# All TargetInit allow retreiving properties with `getproperty`
# from the parent `ConnectedGraphInit` or fields calculated and stored in
# the `TargetInit` storage object.
@inline function Base.getproperty(ti::TargetInit, x::Symbol)
    # Handle all standard properties
    if x === :qᵗ
        # For TargetInit we alter target quality to be just the current target
        return targetquality(ti)[targetconnectedgraphidx(ti)]::Float64
    elseif hasproperty(connectedgraphinit(ti), x)
        return getproperty(connectedgraphinit(ti), x)
    else
        # Defer to `get_or_compute_target` for all other properties
        # We wrap the output in a `ReadOnlyArray` to prevent bugs.
        return get_or_compute_target!(ti, x)
    end
end
# TODO: complete this for all movement modes ?


# `CommonSolve.init`
# Object initialisation methods.
# These all take at least a measures and movement mode specification and a spatial object.
# Spatial object may be a RasterStack or a GridGraph, or another Initialisation object
# from a higher level.

function init(
    m::Union{Measure,MeasureTuple,MeasureNamedTuple}, i::Initialisation, args...; kw...
)
    init(init(m, i), args...; kw...)
end
function init(movement::MovementMode, x::Union{RasterStack,GridGraph}, args...; kw...)
    init(ConScapeProblem(; movement, kw...), x, args...)
end
function init(
    measure::Union{Measure,MeasureTuple,MeasureNamedTuple},
    movement::MovementMode,
    x::Union{RasterStack,GridGraph},
    args...;
    kw...
)
    init(ConScapeProblem(measure; movement, kw...), x, args...)
end
function init(problem::ConScapeProblem, x::Union{RasterStack,GridGraph}; kw...)
    GridGraphInit(problem, x; kw...)
end
function init(
    problem::ConScapeProblem, x::Union{RasterStack,GridGraph}, connectedgraph::Int; 
    finallevel=ConnectedGraphLevel(), kw...
)
    init(GridGraphInit(problem, x; finallevel, kw...), connectedgraph; finallevel)
end
# We don't want to allocate outputs if we work at the target level
function init(
    problem::ConScapeProblem,
    x::Union{RasterStack,GridGraph},
    connectedgraph::Int,
    target::Union{Int,CartesianIndex,TargetID};
    kw...
)
    ggi = GridGraphInit(problem, x; outputs=nothing, kw...)
    init(ggi, connectedgraph, target; finallevel=TargetLevel())
end
init(ggi::GridGraphInit, connectedgraph::Int; kw...) = 
    ConnectedGraphInit(ggi, connectedgraph; kw...)
function init(
    ggi::GridGraphInit, connectedgraph::Int, target::Union{Int,CartesianIndex,TargetID}; 
    kw...
)
    init(ConnectedGraphInit(ggi, connectedgraph; kw...), target)
end
function init(
    gi::ConnectedGraphInit, target::Union{Int,CartesianIndex,TargetID}; kw...
)
    TargetInit(gi, target; kw...)
end
function init(
    m::Union{Measure,MeasureTuple,MeasureNamedTuple}, p::ConScapeProblem, args...; 
    kw...
)
    init(setmeasures(p, m), args...; kw...)
end
function init(
    m::Union{Measure,MeasureTuple,MeasureNamedTuple}, i::Initialisation; 
    finallevel=defaultfinallevel(i)
)
    setmeasures(i, m; finallevel)
end

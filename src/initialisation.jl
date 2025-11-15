const TargetID = @NamedTuple{spatialidx::CartesianIndex{2},gridgraphidx::Int,connectedgraphidx::Int,node::Int}
const SourceID = CartesianIndex{2}

abstract type Initialisation end

movement(i::Initialisation) = movement(problem(i))
costfunction(i::Initialisation) = costfunction(problem(i))
likelihoodfunction(i::Initialisation) = likelihoodfunction(problem(i))
solver(i::Initialisation) = solver(problem(i))
measures(i::Initialisation) = measures(problem(i))
proximity_measure(i::Initialisation) = proximity_measure(problem(i))
distance_transformation(i::Initialisation) = distance_transformation(problem(i))
diagvalue(i::Initialisation) = diagvalue(problem(i))
approx(i::Initialisation) = approx(movement(i))
theta(i::Initialisation) = theta(movement(i))

gridgraph_size(i::Initialisation) = gridgraph_size(gridgraph(i))
target_size(i::Initialisation) = (ntargets(gridgraph(i)),)

Base.size(i::Initialisation, args...) = Base.size(gridgraph(i), args...)
Base.length(i::Initialisation) = Base.length(gridgraph(i))
DimensionalData.dims(i::Initialisation) = dims(gridgraph(i))

"""
    GridGraphInit

Holds multiple grids for the same regions, splitting
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
struct GridGraphInit{P<:ConScapeProblem,G<:GridGraph,SG<:ConnectedGraph,W<:AbstractVector,S<:Dict,O} <: Initialisation
    problem::P
    gridgraph::G
    connectedgraphs::Vector{SG}
    workspaces::Workspaces{W}
    storage::S
    outputs::O
end
GridGraphInit(problem::ConScapeProblem, rast::RasterStack; kw...) =
    GridGraphInit(problem, GridGraph(problem, rast); kw...)
function GridGraphInit(problem::ConScapeProblem, gridgraph::GridGraph; 
    workspaces=nothing, 
    verbose=false, 
    outputlevel=GridGraphLevel(), 
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
    storage = connectedgraph_storage(movement(problem), workspaces)
    # Only allocate outputs if requested at the grid level
    # Otherwise ConnectedGraphInit will do this further down.
    outputs = if outputlevel isa GridGraphLevel
        allocate_output(outputlevel, problem, gridgraph, connectedgraphs)
    end
    # Now create a GridGraphInit with outputs
    return GridGraphInit(problem, gridgraph, connectedgraphs, workspaces, storage, outputs)
end

problem(ggi::GridGraphInit) = ggi.problem
gridgraph(ggi::GridGraphInit) = ggi.gridgraph
connectedgraphs(ggi::GridGraphInit) = ggi.connectedgraphs
workspaces(ggi::GridGraphInit) = ggi.workspaces
outputs(ggi::GridGraphInit) = ggi.outputs
storage(ggi::GridGraphInit) = ggi.storage
nconnectedgraphs(ggi::GridGraphInit) = length(connectedgraphs(ggi))
stepcost(p::GridGraphInit) = stepcost(gridgraph(p))
steplikelihood(p::GridGraphInit) = steplikelihood(gridgraph(p))
sourcequality(p::GridGraphInit) = sourcequality(gridgraph(p))
targetquality(p::GridGraphInit) = targetquality(gridgraph(p))

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
struct ConnectedGraphInit{MM,P<:ConScapeProblem{MM},GG<:GridGraph,SG<:ConnectedGraph,O<:Union{Nothing,NamedTuple,Tuple},W<:AbstractArray,S<:Dict,Pr} <: Initialisation
    problem::P
    gridgraph::GG
    connectedgraph::SG
    outputs::O
    workspaces::Workspaces{W}
    storage::S
    precalculation::Pr
    connectedgraphid::Int
    # Internal constructor enforces:
    # 1. workspace length matches number of sources 
    # 2. storage Dict is empty
    # 3. workspaces are all available for use

    function ConnectedGraphInit(
        problem::P, gridgraph::GG, connectedgraph::SG, outputs::O, workspaces::Workspaces{W}, storage::S, precalculation::Pr, id::Int
    ) where {P<:ConScapeProblem{MM},GG,SG,O,W,S,Pr} where MM
        @assert length(workspaces) == nsources(connectedgraph)
        empty!(storage)
        free!(workspaces)
        new{MM,P,GG,SG,O,W,S,Pr}(problem, gridgraph, connectedgraph, outputs, workspaces, storage, precalculation, id)
    end
end
function ConnectedGraphInit(ggi::GridGraphInit, connectedgraphid::Int; 
    outputlevel=ConnectedGraphLevel(),
    workspaces=workspaces(ggi),
    storage=storage(ggi), 
    kw...
)
    connectedgraph = connectedgraphs(ggi)[connectedgraphid]
    _check_inputs(ggi, connectedgraph)
    workspaces = _allocate_workspaces!(workspaces, problem(ggi), connectedgraph)
    if isnothing(storage) 
        storage = connectedgraph_storage(movement(ggi), workspaces)
    end
    precalculation = connectedgraph_precalculation(problem(ggi), connectedgraph)
    outputs = allocate_output(outputlevel, problem(ggi), gridgraph(ggi), connectedgraph, precalculation)
    return ConnectedGraphInit(problem(ggi), gridgraph(ggi), connectedgraph, outputs, workspaces, storage, precalculation, connectedgraphid)
end

gridgraph(sgi::ConnectedGraphInit) = sgi.gridgraph
connectedgraph(sgi::ConnectedGraphInit) = sgi.connectedgraph
problem(sgi::ConnectedGraphInit) = sgi.problem
workspaces(sgi::ConnectedGraphInit) = sgi.workspaces
outputs(sgi::ConnectedGraphInit) = sgi.outputs
storage(sgi::ConnectedGraphInit) = sgi.storage
precalculation(sgi::ConnectedGraphInit) = sgi.precalculation
connectedgraphid(sgi::ConnectedGraphInit) = sgi.connectedgraphid
nsources(sgi::ConnectedGraphInit) = nsources(connectedgraph(sgi))
ntargets(sgi::ConnectedGraphInit) = ntargets(connectedgraph(sgi))

stepcost(sgi::ConnectedGraphInit) = stepcost(connectedgraph(sgi))
steplikelihood(sgi::ConnectedGraphInit) = steplikelihood(connectedgraph(sgi))
sourcequality(sgi::ConnectedGraphInit) = sourcequality(connectedgraph(sgi))
targetquality(sgi::ConnectedGraphInit) = targetquality(connectedgraph(sgi))
sourceids(sgi::ConnectedGraphInit) = sourceids(connectedgraph(sgi))
targetids(sgi::ConnectedGraphInit) = targetids(connectedgraph(sgi))
connectedgraph_size(sgi::ConnectedGraphInit) = connectedgraph_size(connectedgraph(sgi))

connectedgraph_storage(::MovementMode, ::Workspaces{W}) where W<:AbstractArray{T} where T = 
    Dict{Symbol,ReadOnlyArray{T,1,W}}()
# Need to store the Woodbury matrix 
connectedgraph_storage(::RandomWalk, ::Workspaces{W}) where W<:AbstractArray{T} where T = 
    Dict{Symbol,Any}()

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
struct TargetInit{MM,SI<:ConnectedGraphInit{MM}} <: Initialisation
    connectedgraphinit::SI
    target::TargetID
    function TargetInit(sgi::SI, target::TargetID) where SI<:ConnectedGraphInit{MM} where MM
        @assert (length(workspaces(sgi)) == nsources(connectedgraph(sgi)) == connectedgraph_size(sgi)[1]) 
        free!(workspaces(sgi))
        empty!(storage(sgi))
        new{MM,SI}(sgi, target)
    end
end
TargetInit(sgi::ConnectedGraphInit, target::Int) = TargetInit(sgi, targetids(sgi)[target])
function TargetInit(sgi::ConnectedGraphInit, target::CartesianIndex)
    i = findfirst(t -> t.spatialidx == target, targetids(sgi))
    isnothing(i) && throw(ArgumentError("target indices $target are not part of this network"))
    TargetInit(sgi, targetids(sgi)[i])
end

connectedgraphinit(ti::TargetInit) = getfield(ti, :connectedgraphinit)
target(ti::TargetInit) = getfield(ti, :target)

outputs(ti::TargetInit) = outputs(connectedgraphinit(ti))
gridgraph(ti::TargetInit) = gridgraph(connectedgraphinit(ti))
connectedgraph(ti::TargetInit) = connectedgraph(connectedgraphinit(ti))
problem(ti::TargetInit) = problem(connectedgraphinit(ti))
storage(ti::TargetInit) = storage(connectedgraphinit(ti))
workspaces(ti::TargetInit) = workspaces(connectedgraphinit(ti))
stepcost(p::TargetInit) = stepcost(connectedgraph(p))
steplikelihood(p::TargetInit) = steplikelihood(connectedgraph(p))
sourcequality(p::TargetInit) = sourcequality(connectedgraph(p))
targetquality(p::TargetInit) = targetquality(connectedgraph(p))
sourceids(p::TargetInit) = sourceids(connectedgraph(p))
targetids(p::TargetInit) = targetids(connectedgraph(p))
connectedgraph_size(i::TargetInit) = connectedgraph_size(connectedgraphinit(i))
precalculation(i::TargetInit) = precalculation(connectedgraphinit(i))

# All TargetInit allow retreiving proberties with `getproperty`
# from the parent `ConnectedGraphInit` or calculated and stored in 
# the `TargetInit`
@inline function Base.getproperty(ti::TargetInit, x::Symbol)
    if x === :workspace
        return take!(workspaces(ti))
    elseif x === :θ 
        return theta(ti)
    elseif x === :qᵗ
        return targetquality(ti)[target(ti).connectedgraphidx]
    elseif x === :qˢ
        return sourcequality(ti)
    elseif x === :A
        return steplikelihood(ti)
    elseif x === :C
        return stepcost(ti)
    elseif hasproperty(connectedgraphinit(ti).precalculation, x)
        return Base.getproperty(connectedgraphinit(ti).precalculation, x)
    end
    # Defer to `get_or_compute` for all other properties
    # We wrap the output in a `ReadOnlyArray` to prevent bugs.
    return get_or_compute!(ti, x)
end

# `CommonSolve.init`

init(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, i::Initialisation, args...; kw...) = 
    init(init(m, i), args...; kw...)
init(movement::MovementMode, x::Union{RasterStack,GridGraph}, args...; kw...) = 
    init(ConScapeProblem(; movement, kw...), x, args...)
init(measure::Union{Measure,MeasureTuple,MeasureNamedTuple}, 
    movement::MovementMode, 
    x::Union{RasterStack,GridGraph}, 
    args...; 
    kw...
) = init(ConScapeProblem(measure; movement, kw...), x, args...)
init(problem::ConScapeProblem, x::Union{RasterStack,GridGraph}; kw...) = 
    GridGraphInit(problem, x; kw...)
init(problem::ConScapeProblem, x::Union{RasterStack,GridGraph}, connectedgraph::Int; outputlevel=ConnectedGraphLevel(), kw...) = 
    init(GridGraphInit(problem, x; outputlevel, kw...), connectedgraph; outputlevel)
# We don't want to allocate outputs if we work at the target level
init(problem::ConScapeProblem, x::Union{RasterStack,GridGraph}, connectedgraph::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) = 
    init(GridGraphInit(problem, x; outputs=nothing, kw...), connectedgraph, target; outputlevel=TargetLevel())
init(ggi::GridGraphInit, connectedgraph::Int; kw...) = ConnectedGraphInit(ggi, connectedgraph; kw...)
init(ggi::GridGraphInit, connectedgraph::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) = 
    init(ConnectedGraphInit(ggi, connectedgraph; kw...), target)
init(gi::ConnectedGraphInit, target::Union{Int,CartesianIndex,TargetID}) = TargetInit(gi, target)
init(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, p::ConScapeProblem, args...; kw...) =
    init(setmeasures(p, m), args...; kw...)
init(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, i::Initialisation; outputlevel=defaultoutputlevel(i)) =
    setmeasures(i, m; outputlevel)

setmeasures(p::ConScapeProblem, m::Measure) = setmeasures(p, NamedTuple{(Symbol(m),)}((m,)))
setmeasures(p::ConScapeProblem, measures::Union{MeasureTuple,MeasureNamedTuple}) = 
    ConstructionBase.setproperties(p, (; measures))
function setmeasures(i::Union{GridGraphInit,ConnectedGraphInit}, m; outputlevel)
    problem = setmeasures(ConScape.problem(i), m)
    outputs = map(measures(problem)) do m
        allocate_output(outputlevel, m, i)
    end
    return ConstructionBase.setproperties(i, (; problem, outputs))
end
function setmeasures(i::TargetInit, m; outputlevel)
    connectedgraphinit = setmeasures(connectedgraphinit(i), m; outputlevel)
    return ConstructionBase.setproperties(i, (; connectedgraphinit))
end


defaultoutputlevel(::GridGraphInit) = GridGraphLevel()
defaultoutputlevel(::ConnectedGraphInit) = ConnectedGraphLevel()
defaultoutputlevel(::TargetInit) = TargetLevel()

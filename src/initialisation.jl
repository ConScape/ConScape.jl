const TargetID = @NamedTuple{spatialidx::CartesianIndex{2},graphidx::Int,subgraphidx::Int,node::Int}
const SourceID = CartesianIndex{2}

abstract type Initialisation end

movement(p::Initialisation) = movement(problem(p))
costfunction(p::Initialisation) = costfunction(movement(p))
solver(p::Initialisation) = solver(problem(p))
measures(p::Initialisation) = measures(problem(p))
proximity_measure(p::Initialisation) = proximity_measure(problem(p))
distance_transformation(p::Initialisation) = distance_transformation(problem(p))
diagvalue(p::Initialisation) = diagvalue(problem(p))
approx(p::Initialisation) = approx(movement(p))
theta(p::Initialisation) = theta(movement(p))

nsources(p::Initialisation) = nsources(gridgraph(p))
ntargets(p::Initialisation) = ntargets(gridgraph(p))
sparse_size(p::Initialisation) = nsources(p), ntargets(p) 

Base.size(p::Initialisation, args...) = Base.size(gridgraph(p), args...)
Base.length(p::Initialisation) = Base.length(gridgraph(p))
DimensionalData.dims(p::Initialisation) = dims(gridgraph(p))

"""
    GridGraph(size::Tuple{Int,Int}; kw...)

Construct a `GridGraph` from an `transitionlikelihood` matrix of type `SparseMatrixCSC`. 

# Keywords

- `transitionlikelihood`: nothing
- `qualities::Matrix`: ones(nrows, ncols)
- `source_qualities::Matrix`: qualities
- `target_qualities::AbstractMatrix`: qualities
- `costfunction`: `MinusLog()` by default.
- `transitioncost`: optionally specify a sparse cost matrix. 
    By default it is calculated from `costfunction.(transitionlikelihood)`
- `prune`: if the likelihood and cost matrices will be pruned 
    to exclude unreachable nodes. `true` by default.

It is possible to also supply matrices of `source_qualities` and `target_qualities` as well as

Alternatively, it is possible to supply a matrix to `costs` directly. If `prune=true` (the default), 
"""
struct GridGraph{
    C<:Union{AbstractMatrix,Nothing},
    L<:Union{AbstractMatrix,Nothing},
    SQ<:AbstractMatrix,
    TQ<:AbstractMatrix,
    D<:Union{Tuple,Nothing}
} <: Initialisation
    transitioncost::C
    transitionlikelihood::L
    sourcequality::SQ
    targetquality::TQ
    dims::D
end
function GridGraph(;
    quality::Union{AbstractMatrix,Nothing}=nothing,
    sourcequality::AbstractMatrix=quality,
    targetquality::AbstractMatrix=quality,
    cost=nothing,
    likelihood=nothing,
    costfunction::Union{Function,Transformation,Nothing}=MinusLog(),
    likelihoodfunction::Union{Function,Transformation,Nothing}=nothing,
    grain=nothing,
    kw...
)
    isnothing(cost) && isnothing(likelihood) && 
        throw(ArgumentError("At least one of `cost` and `likelihood` must be specified"))
    transitionlikelihood = if !isnothing(likelihood) 
        graph_matrix_from_raster(likelihood; input_type=Likelihood(), kw...)
    end
    transitioncost = if !isnothing(cost)
        graph_matrix_from_raster(cost; input_type=Cost(), kw...) 
    end
    if isnothing(transitionlikelihood) && !isnothing(likelihoodfunction)
        transitionlikelihood = mapnz(likelihoodfunction, transitioncost)
    end
    if isnothing(transitioncost) && !isnothing(costfunction)
        transitioncost = mapnz(costfunction, transitionlikelihood)
    end

    # This is too expensive to calculate for small target grids
    # if check
    #     if any(t -> t < 0, nonzeros(transitioncost))
    #         throw(ArgumentError("The cost graph can have only non-negative edge weights. Perhaps you should change the cost function?"))
    #     end
    #     cost_digraph = SimpleDiGraph(transitioncost)
    #     likelihood_digraph = SimpleDiGraph(transitionlikelihood)

    #     if ne(difference(cost_digraph, likelihood_digraph)) > 0
    #         throw(ArgumentError("cost graph contains edges not present in the likelihood graph"))
    #     end
    # end

    if !isnothing(transitionlikelihood) && prod(size(sourcequality)) != LinearAlgebra.checksquare(transitionlikelihood)
        throw(ArgumentError("grid size $size is incompatible with size of transitionlikelihood matrix ($n, $n)"))
    end
    if !isnothing(transitioncost) && prod(size(sourcequality)) != LinearAlgebra.checksquare(transitioncost)
        throw(ArgumentError("grid size $size is incompatible with size of transitioncost matrix ($n, $n)"))
    end

    # Subset of source_ids with valid quality
    if !isnothing(grain)
        targetquality = coarse_graining(targetquality, grain)
    end
    return GridGraph(
        transitioncost,
        transitionlikelihood,
        _prepare_qualities(sourcequality),
        _prepare_qualities(targetquality),
        dims(sourcequality),
    )
end
function GridGraph(rast::RasterStack;
    cost=_get_cost(rast),
    likelihood=_get_likelihood(rast),
    sourcequality=_get_sourcequality(rast),
    targetquality=_get_targetquality(rast),
    kw...
)
    GridGraph(; likelihood, cost, sourcequality, targetquality, kw...)
end
function GridGraph(p::AbstractProblem, rast::RasterStack; kw...)
    GridGraph(rast; 
        grain=grain(p), 
        costfunction=costfunction(p), 
        likelihoodfunction=likelihoodfunction(p), 
        neighbors=neighbors(p),
        transition_weight=transition_weight(p),
        kw...
    )
end

transitionlikelihood(g::GridGraph) = g.transitionlikelihood
transitioncost(g::GridGraph) = g.transitioncost
sourcequality(g::GridGraph) = g.sourcequality
targetquality(g::GridGraph) = g.targetquality
sourceids(g::GridGraph) = vec(CartesianIndices(sourcequality(g)))
nsources(g::GridGraph) = length(g)
# TODO is this a memory problem for custom use?
ntargets(g::GridGraph) = nsources(g) 

Base.size(g::GridGraph, args...) = size(sourcequality(g), args...)
Base.length(g::GridGraph) = length(sourcequality(g))
Base.show(io::IO, ::MIME"text/plain", g::GridGraph) =
    print(io, "GridGraph of size ", size(g))

DimensionalData.dims(g::GridGraph) = g.dims

struct ConnectedGraph{C<:Union{AbstractMatrix,Nothing},L<:Union{AbstractMatrix,Nothing},SQ<:AbstractVector,TQ<:AbstractVector,SI,TI} <: Initialisation
    transitioncost::C
    transitionlikelihood::L
    sourcequality::SQ
    targetquality::TQ
    sourceids::SI
    targetids::TI
end

transitioncost(cg::ConnectedGraph) = cg.transitioncost
transitionlikelihood(cg::ConnectedGraph) = cg.transitionlikelihood
sourcequality(cg::ConnectedGraph) = cg.sourcequality
targetquality(cg::ConnectedGraph) = cg.targetquality
sourceids(cg::ConnectedGraph) = cg.sourceids
targetids(cg::ConnectedGraph) = cg.targetids
nsources(cg::ConnectedGraph) = length(sourceids(cg))
ntargets(cg::ConnectedGraph) = length(targetids(cg))


"""
    ProblemInit

Holds multiple grids for the same regions, splitting
them into subgraphs.

Also allocates and holds workspaces and outputs,
as `ProblemInit` is the level at which
whole spatial problems are solved. 

`solve` on `ProblemInit` iterates over 
subgraphs, generating a `SubgraphInit` for each
and solving it into the same output object.

## Example

To construct a `ProblemInit` :

```julia
pi = init(problem, rast)
```

To solve measures in the `Problem`:

```julia
results = solve(pi)
```

To solve arbitrary measures:

```julia
ec = solve(ExpectedCost(), pi)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), pi)
```
"""
struct ProblemInit{P<:Problem,G<:GridGraph,SG<:ConnectedGraph,W<:AbstractVector,S<:Dict,O} <: Initialisation
    problem::P
    gridgraph::G
    subgraphs::Vector{SG}
    workspaces::Workspaces{W}
    storage::S
    outputs::O
end
ProblemInit(problem::Problem, rast::RasterStack; kw...) =
    ProblemInit(problem, GridGraph(problem, rast); kw...)
function ProblemInit(problem::Problem, gridgraph::GridGraph; 
    workspaces=nothing, verbose=false, outputs=true,
)
    _check_inputs(problem, gridgraph)
    subgraphs = split_subgraphs(gridgraph; 
        costfunction=costfunction(problem),
        likelihoodfunction=likelihoodfunction(problem),
    )
    workspaces = if length(subgraphs) > 0
        _allocate_workspaces!(workspaces, problem, first(subgraphs))
    else
        Workspaces(0, 0)
    end
    storage = subgraph_storage(movement(problem), workspaces)
    # Create a ProblemInit without outputs
    pi = ProblemInit(problem, gridgraph, subgraphs, workspaces, storage, nothing)
    if outputs isa Bool && outputs
        # Generate outputs for each graph measure, the first subgraph is the largest
        outputs = allocate_output(problem, pi)
    end
    # Now create a ProblemInit with outputs
    return ProblemInit(problem, gridgraph, subgraphs, workspaces, storage, outputs)
end

problem(pi::ProblemInit) = pi.problem
gridgraph(pi::ProblemInit) = pi.gridgraph
subgraphs(pi::ProblemInit) = pi.subgraphs
workspaces(pi::ProblemInit) = pi.workspaces
outputs(pi::ProblemInit) = pi.outputs
storage(pi::ProblemInit) = pi.storage
nsubgraphs(pi::ProblemInit) = length(subgraphs(pi))
transitioncost(p::ProblemInit) = transitioncost(gridraph(p))
transitionlikelihood(p::ProblemInit) = transitionlikelihood(gridgraph(p))
sourcequality(p::ProblemInit) = sourcequality(gridgraph(p))
targetquality(p::ProblemInit) = targetquality(gridgraph(p))
sourceids(p::ProblemInit) = sourceids(gridgraph(p))
targetids(p::ProblemInit) = targetids(gridgraph(p))

_check_inputs(x, init) = _check_inputs(movement(x), init)
function _check_inputs(::RSP, g)
    isnothing(transitioncost(g)) && throw(ArgumentError("GridGraph has no transitioncost for RSP"))
    isnothing(transitionlikelihood(g)) && throw(ArgumentError("GridGraph has no transitionlikelihood for RSP"))
end
function _check_inputs(::LCP, g)
    isnothing(transitioncost(g)) && throw(ArgumentError("GridGraph has no transitioncost for LCP"))
end
function _check_inputs(::RandomWalk, g)
    isnothing(transitionlikelihood(g)) && throw(ArgumentError("GridGraph has no transitionlikelihood for LCP"))
end
_check_inputs(::Euclidean, g) = nothing

"""
    SubgraphInit

Precalculated variables and outputs buckets used 
in `solve` and `compute` methods.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables such as sparse factorization once for all targets.

## Example

To construct a `SubgraphInit` :

```julia
probleminit = init(problem, rast)
si = init(probleminit, 1)
````

To solve measures in the `Problem` for this subgraph:

```julia
results = solve(si)
```

To solve arbitrary measures for this subgraph:

```julia
ec = solve(ExpectedCost(), si)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), si)
```
"""
struct SubgraphInit{MM,P<:Problem{MM},GG<:GridGraph,SG<:ConnectedGraph,O<:Union{Nothing,NamedTuple,Tuple},W<:AbstractArray,S<:Dict,Pr} <: Initialisation
    problem::P
    gridgraph::GG
    subgraph::SG
    outputs::O
    workspaces::Workspaces{W}
    storage::S
    precalculation::Pr
    function SubgraphInit(
        problem::P, gridgraph::GG, subgraph::SG, outputs::O, workspaces::Workspaces{W}, storage::S, precalculation::Pr
    ) where {P<:Problem{MM},GG,SG,O,W,S,Pr} where MM
        @assert length(workspaces) == nsources(subgraph)
        free!(workspaces)
        empty!(storage)
        new{MM,P,GG,SG,O,W,S,Pr}(problem, gridgraph, subgraph, outputs, workspaces, storage, precalculation)
    end
end
function SubgraphInit(problem::Problem, gridgraph::GridGraph, subgraph::ConnectedGraph;
    outputs=allocate_output(problem, subgraph),
    workspaces=nothing, 
    storage=nothing,
    verbose=false,
)
    _check_inputs(problem, subgraph)
    workspaces = _allocate_workspaces!(workspaces, problem, subgraph)
    if isnothing(storage) 
        storage = subgraph_storage(movement(problem), workspaces)
    end
    precalculation = subgraph_precalculation(problem, subgraph)
    return SubgraphInit(problem, gridgraph, subgraph, outputs, workspaces, storage, precalculation)
end
function SubgraphInit(pi::ProblemInit, subgraph_num::Int; 
    workspaces=workspaces(pi),
    outputs=outputs(pi), 
    storage=storage(pi), 
    kw...
)
    SubgraphInit(problem(pi), gridgraph(pi), subgraphs(pi)[subgraph_num]; 
        workspaces, outputs, storage, kw...
    )
end

gridgraph(si::SubgraphInit) = si.gridgraph
subgraph(si::SubgraphInit) = si.subgraph
problem(si::SubgraphInit) = si.problem
workspaces(si::SubgraphInit) = si.workspaces
outputs(si::SubgraphInit) = si.outputs
storage(si::SubgraphInit) = si.storage
nsources(si::SubgraphInit) = nsources(subgraph(si))
ntargets(si::SubgraphInit) = ntargets(subgraph(si))

transitioncost(p::SubgraphInit) = transitioncost(subgraph(p))
transitionlikelihood(p::SubgraphInit) = transitionlikelihood(subgraph(p))
sourcequality(p::SubgraphInit) = sourcequality(subgraph(p))
targetquality(p::SubgraphInit) = targetquality(subgraph(p))
sourceids(p::SubgraphInit) = sourceids(subgraph(p))
targetids(p::SubgraphInit) = targetids(subgraph(p))

subgraph_storage(::MovementMode, ::Workspaces{W}) where W<:AbstractArray{T} where T = 
    Dict{Symbol,ReadOnlyArray{T,1,W}}()
# Need to store the Woodbury matrix 
subgraph_storage(::RandomWalk, ::Workspaces{W}) where W<:AbstractArray{T} where T = 
    Dict{Symbol,Any}()

"""
    TargetInit

Abstract type for precalculated variables at the level of single targets.

Dense vector variables like `Z` (fundamental matrix) are generated 
on demand in `getproperty` (e.g. `tp.Z`) and stored for subsequent requests.

These variables use preallocated [`Workspaces`](@ref) to avoid allocations.

Variables from the parent `SubgraphInit` can also be accessed with
`getpropery`, e.g. `tp.W` returns a sparse matrix calculated for all targets.

Stores outputs and lazily calculated variables for use in `RandomisedShortestPath`-based measures.

Varables can be accessed with `getproperty`: `rsp_tp.Z`.

From the parent `ConnectedGraph`, the available variables are:
`qᵗ`, `qˢ`, `L`, `C`

From the parent `SubgraphInit`, the available variables are:

`P`, `W`, `IW`, `CW`, `IW_factorization`, `IW_adj`, `IW_adj_factorization`,

For target dense vectors:

`Z`, `Zⁱ`, `Zrows`, `Q`, `K`, `M`.

## Example

To construct a `TargetInit` :

```julia
probleminit = init(problem, rast)
subgraph = 1
subgraphinit = init(probleminit, subgraph)
target_idx = 7 
ti = init(subgraphinit, target_idx)
````

To solve measures in the `Problem` for this target:

```julia
results = solve(ti)
```

To solve arbitrary measures for this target:

```julia
ec = solve(ExpectedCost(), ti)
ec, ch = solve((ExpectedCost(), ConnectedHabitat()), ti)
```
"""
struct TargetInit{MM,SI<:SubgraphInit{MM}} <: Initialisation
    subgraphinit::SI
    target::TargetID
    function TargetInit(gi::GI, target::TargetID) where GI<:SubgraphInit{MM} where MM
        @assert (length(workspaces(gi)) == nsources(gi) == sparse_size(gi)[1]) 
        free!(workspaces(gi))
        empty!(storage(gi))
        new{MM,GI}(gi, target)
    end
end
TargetInit(gi::SubgraphInit, target::Int) = TargetInit(gi, targetids(gi)[target])
function TargetInit(gi::SubgraphInit, target::CartesianIndex)
    i = findfirst(t -> t.spatialidx == target, targetids(gi))
    isnothing(i) && throw(ArgumentError("target indices $target are not part of this network"))
    TargetInit(gi, targetids(gi)[i])
end

subgraphinit(ti::TargetInit) = getfield(ti, :subgraphinit)
target(ti::TargetInit) = getfield(ti, :target)

outputs(ti::TargetInit) = outputs(subgraphinit(ti))
gridgraph(ti::TargetInit) = gridgraph(subgraphinit(ti))
subgraph(ti::TargetInit) = subgraph(subgraphinit(ti))
problem(ti::TargetInit) = problem(subgraphinit(ti))
storage(ti::TargetInit) = storage(subgraphinit(ti))
workspaces(ti::TargetInit) = workspaces(subgraphinit(ti))
transitioncost(p::TargetInit) = transitioncost(subgraph(p))
transitionlikelihood(p::TargetInit) = transitionlikelihood(subgraph(p))
sourcequality(p::TargetInit) = sourcequality(subgraph(p))
targetquality(p::TargetInit) = targetquality(subgraph(p))
sourceids(p::TargetInit) = sourceids(subgraph(p))
targetids(p::TargetInit) = targetids(subgraph(p))

# All TargetInit allow retreiving proberties with `getproperty`
# from the parent `SubgraphInit` or calculated and stored in 
# the `TargetInit`
@inline function Base.getproperty(ti::TargetInit, x::Symbol)
    if x === :workspace
        return take!(workspaces(ti))
    elseif x === :θ 
        return theta(ti)
    elseif x === :qᵗ
        return targetquality(ti)[target(ti).subgraphidx]
    elseif x === :qˢ
        return sourcequality(ti)
    elseif x === :A
        return transitionlikelihood(ti)
    elseif x === :C
        return transitioncost(ti)
    elseif hasproperty(subgraphinit(ti).precalculation, x)
        return Base.getproperty(subgraphinit(ti).precalculation, x)
    end
    # Defer to `get_or_compute` for all other properties
    # We wrap the output in a `ReadOnlyArray` to prevent bugs.
    return get_or_compute!(ti, x)
end

function subgraph_precalculation(problem::Problem{<:RSP}, graph::ConnectedGraph)
    P, A_rowsums = _transitionprobability(transitionlikelihood(graph))
    W = _substochasticmatrix(movement(problem), P, transitioncost(graph))
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    Aⁱ = mapnz(inv, transitionlikelihood(graph))
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
    CW = transitioncost(graph) .* W

    return (; P, W, IW, IW_adj, CW, IW_factorization, IW_adj_factorization, Aⁱ, A_rowsums)
end
function subgraph_precalculation(::Problem{<:LCP}, graph::ConnectedGraph)
    P, L_rowsums = _transitionprobability(transitionlikelihood(graph))
    # TODO: use a raster based shortest path algorithm from Geomorphometry.jl
    # disjkstra is especially slow due to allocations,
    # searchsorted for index lookups, and Dict getindex/setindex!.
    cost_weighted_digraph = SimpleWeightedDiGraph(transitioncost(graph))
    dsp1 = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, 1)
    parents = dsp1.parents
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in 1:length(parents)]
    (; P, L_rowsums, cost_weighted_digraph, path_allocs)
end
function subgraph_precalculation(problem::Problem{<:RandomWalk}, graph::ConnectedGraph)
    P, L_rowsums = _probabilitymatrix(transitionlikelihood(graph))
    Lⁱ = mapnz(inv, transitionlikelihood(graph))
    PC = P .* transitioncost(graph)
    PC_rowsums = sum(PC; dims=2)
    IP = I - P
    IP_factorization = init(solver(problem), IP)
    return (; Lⁱ, L_rowsums, P, PC, PC_rowsums, IP, IP_factorization)
end
function subgraph_precalculation(::Problem{<:Euclidean}, ::ConnectedGraph)
    (;)
end

# Variable generation for SubgraphInit
function _transitionprobability(L::SparseMatrixCSC)
    source_sums = vec(sum(L, dims=2))
    source_scaling = inv.(source_sums)
    P = Diagonal(source_scaling) * L
    return P, source_sums
end
# Substochastic
function _substochasticmatrix(rsp::RSP, P::SparseMatrixCSC, C::SparseMatrixCSC)
    @assert LinearAlgebra.checksquare(C) == LinearAlgebra.checksquare(P)
    W = P .* exp.(-theta(rsp) .* C)
    replace!(W.nzval, NaN => 0.0)
    return W
end

# function _check_z(ti::TargetInit{<:RSP})
#     # Check that values in Z are not too small
#     # TODO: does this make sense for single targets
#     if check(ti) && minimum(ti.Z) * minimum(nonzeros(ti.CW)) == 0
#         @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
#     end
# end

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

init(movement::MovementMode, x::Union{RasterStack,GridGraph}, args...; kw...) = 
    init(Problem(; movement, kw...), x, args...)
init(measure::Union{Measure,MeasureTuple,MeasureNamedTuple}, 
    movement::MovementMode, 
    x::Union{RasterStack,GridGraph}, 
    args...; 
    kw...
) = init(Problem(measure; movement, kw...), x, args...)
init(problem::Problem, x::Union{RasterStack,GridGraph}; kw...) = 
    ProblemInit(problem, x; kw...)
init(problem::Problem, x::Union{RasterStack,GridGraph}, subgraph::Int; kw...) = 
    init(ProblemInit(problem, x; kw...), subgraph)
# We don't want to allocate outputs if we work at the target level
init(problem::Problem, x::Union{RasterStack,GridGraph}, subgraph::Int, target::Union{Int,CartesianIndex,TargetID}; outputs=nothing, kw...) = 
    init(ProblemInit(problem, x; outputs, kw...), subgraph, target)
init(pi::ProblemInit, subgraph::Int; kw...) = SubgraphInit(pi, subgraph; kw...)
init(pi::ProblemInit, subgraph::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) = 
    init(SubgraphInit(pi, subgraph; kw...), target)
init(gi::SubgraphInit, target::Union{Int,CartesianIndex,TargetID}) = TargetInit(gi, target)

solve(p::Problem, x::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve(init(p, x; kw...), args...)
solve(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, p::Problem, input::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve(m, init(p, input; kw...), args...)
solve(measures::MeasureTupleOrNamedTuple, x::Union{GridGraph,RasterStack}, args...; verbose=false, kw...) = 
    solve(Problem(measures; kw...), x, args...; verbose)
function solve(
    m::MeasureTupleOrNamedTuple, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; 
    verbose=false, kw...
) 
    solve(Problem(m; movement, kw...), x, args...; verbose)
end
solve(m::Measure, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; verbose=false, kw...) =
    only(values(solve(Problem((m,); movement, kw...), x, args...; verbose)))
# Allow solving all the levels of precalculated object with specific measures
solve(p::Initialisation, args...; kw...) = solve(measures(p), p, args...; kw...)
solve(measure::Measure, init::Initialisation, args...; kw...) =
    only(values(solve((measure,), init, args...; kw...)))
function solve(measures::MeasureTupleOrNamedTuple, pi::ProblemInit; 
    outputs=_maybe_new_outputs(measures, pi), kw...
)
    # Loop over unnconnected subgraphs (there may be only one)
    for i in eachindex(subgraphs(pi))
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        solve(measures, init(pi, i; outputs))
    end
    return _maybe_raster_outputs(measures, outputs, pi)
end
solve(measures::MeasureTupleOrNamedTuple, pi::ProblemInit, i::Int; kw...) =
    solve(measures, init(pi, i; kw...))
solve(measures::MeasureTupleOrNamedTuple, pi::ProblemInit, i::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) =
    solve(measures, init(pi, i, target; outputs=nothing, kw...))
function solve(measures::MeasureTupleOrNamedTuple, si::SubgraphInit; 
    outputs=_maybe_new_outputs(measures, si), kw...
)
    # Then loop over targets
    for target_id in targetids(si)
        # Precalculate for this target and graph measures
        solve(measures, init(si, target_id))
    end
    return _maybe_raster_outputs(measures, outputs, si)
end
solve(measures::MeasureTupleOrNamedTuple, gi::SubgraphInit, i::Int; kw...) =
    solve(measures, init(gi, targetids(gi)[i]); kw...)
function solve(measures::MeasureTupleOrNamedTuple, ti::TargetInit;
    outputs=outputs(ti),
)
    # Allocate target vectors rather than matrices
    outputs1 = if isnothing(outputs)
        allocate_target_output(measures, ti)
    else
        outputs
    end
    # Store outputs
    results = map(measures, outputs1) do measure, output
        # Dont compute the same measure multiple times
        v = get_or_compute!(ti, measure)
        # Write values to output object
        update_output!(output, measure, v, ti)
    end
    if isnothing(outputs)
        return _maybe_raster_outputs(measures, outputs1, ti)
    else
        return results
    end
end
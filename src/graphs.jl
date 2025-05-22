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
}
    transitioncost::C
    transitionlikelihood::L
    sourcequality::SQ
    targetquality::TQ
    dims::D
end
function GridGraph(;
    quality::Union{AbstractMatrix,Nothing}=nothing,
    sourcequality::Union{AbstractMatrix,Nothing}=nothing,
    targetquality::Union{AbstractMatrix,Nothing}=nothing,
    cost=nothing,
    likelihood=nothing,
    transitioncost=nothing,
    transitionlikelihood=nothing,
    costfunction::Union{Function,Transformation,Nothing}=MinusLog(),
    likelihoodfunction::Union{Function,Transformation,Nothing}=nothing,
    grain=nothing,
    kw...
)
    sourcequality = if isnothing(sourcequality)
        isnothing(quality) && throw(ArgumentError("At least one of `quality` and `sourcequality` must be specified"))
        quality
    else
        sourcequality
    end::AbstractMatrix
    targetquality = if isnothing(targetquality)
        isnothing(quality) ? sourcequality : quality
    else
        targetquality
    end::AbstractMatrix

    isnothing(transitioncost) && isnothing(transitionlikelihood) && 
        isnothing(cost) && isnothing(likelihood) && 
            throw(ArgumentError("At least one of `cost` and `likelihood` must be specified"))
    if !isnothing(likelihood) 
        transitionlikelihood = graph_matrix_from_raster(likelihood; input_type=Likelihood(), kw...)
    end
    if !isnothing(cost)
        transitioncost = graph_matrix_from_raster(cost; input_type=Cost(), kw...) 
    end
    if isnothing(transitionlikelihood) && !isnothing(transitioncost) && !isnothing(likelihoodfunction)
        transitionlikelihood = mapnz(likelihoodfunction, transitioncost)
    end
    if isnothing(transitioncost) && !isnothing(transitionlikelihood) && !isnothing(costfunction)
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

    if !isnothing(transitionlikelihood) && prod(size(sourcequality)) != (n = LinearAlgebra.checksquare(transitionlikelihood))
        throw(ArgumentError("quality size $(length(sourcequality)) is incompatible with size of transitionlikelihood matrix ($n, $n)"))
    end
    if !isnothing(transitioncost) && prod(size(sourcequality)) != (n = LinearAlgebra.checksquare(transitioncost))
        throw(ArgumentError("quality size $size is incompatible with size of transitioncost matrix ($n, $n)"))
    end

    # TODO check exact indices of transitioncost and transitionlikelihood match

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
gridgraph_size(gg::GridGraph) = (nsources(gg), ntargets(gg))

Base.size(g::GridGraph, args...) = size(sourcequality(g), args...)
Base.length(g::GridGraph) = length(sourcequality(g))
Base.show(io::IO, ::MIME"text/plain", g::GridGraph) =
    print(io, "GridGraph of size ", size(g))

DimensionalData.dims(g::GridGraph) = g.dims

struct ConnectedGraph{
    C<:Union{AbstractMatrix,Nothing},
    L<:Union{AbstractMatrix,Nothing},
    SQ<:AbstractVector,
    TQ<:AbstractVector,
    SI<:AbstractVector,
    TI<:AbstractVector,
}
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
connectedgraph_size(cg::ConnectedGraph) = (nsources(cg), ntargets(cg))

function _split_connected_graphs(g::GridGraph;
    costfunction=nothing, likelihoodfunction=nothing,
)
    spatialidxs = vec(CartesianIndices(size(g)))
    targetids = _target_ids(targetquality(g), spatialidxs)
    # Convert cost matrix to graph, todo: is `permute=false` needed
    graph = SimpleWeightedDiGraph(
        (isnothing(transitioncost(g)) ? transitionlikelihood(g) : transitioncost(g))::AbstractMatrix; 
        permute=false
    )

    # Find the connected subgraphs
    scc = Graphs.strongly_connected_components(graph)

    # Keep all connected subgraphs that contain target nodes
    subgraphs_with_targets = map(scc) do c
        length(c) > 1 && any(t -> t.node in c, targetids)
    end

    # Sort subgraphs by number of nodes
    connected_subgraphs = sort!(scc[subgraphs_with_targets]; by=length, rev=true)

    # Return a Vector of Grids for each connected subgraph
    return map(connected_subgraphs) do scci
        # Sort subgraph indices
        sort!(scci)

        # Get permeability matrices for the subgraph 
        transcost = if !isnothing(transitioncost(g))
            transitioncost(g)[scci, scci]
        end
        translikelihood = if !isnothing(transitionlikelihood(g))
            transitionlikelihood(g)[scci, scci]
        end
        if isnothing(transcost) && !isnothing(costfunction)
            transcost = mapnz(costfunction, transitionlikelihood(g))
        end
        if isnothing(translikelihood) && !isnothing(likelihoodfunction) 
            translikelihood = mapnz(likelihoodfunction, transitioncost(g))
        end

        # Get new source and target ids for subgraph
        sourceidxs = view(spatialidxs, scci)
        targets = _target_ids(targetquality(g), sourceidxs, spatialidxs)

        # Get source and target quality vectors for subgraph
        sourcequality_vector = view(sourcequality(g), sourceidxs)
        targetquality_vector = [targetquality(g)[i.spatialidx] for i in targets]

        # Return a ConnectedGraph
        ConnectedGraph(
            transcost,
            translikelihood,
            sourcequality_vector,
            targetquality_vector,
            sourceidxs,
            targets,
        )
    end
end


# Compute a vector of the cartesian indices of nonzero target qualities and
# the corresponding node id corresponding to the indices
_target_spatial_ids(target_qauality::AbstractMatrix, source_spatial_ids::AbstractVector) = 
    source_spatial_ids
_target_spatial_ids(target_quality::Raster, source_spatial_ids::AbstractVector) = 
    _target_spatial_ids(parent(target_quality), source_spatial_ids)
function _target_spatial_ids(target_quality::SparseMatrixCSC, source_spatial_ids::AbstractVector)
    is, js, _ = findnz(target_quality)
    return intersect!(CartesianIndex.(is, js), source_spatial_ids)
end

function _target_ids(
    target_quality_spatial::AbstractMatrix, 
    source_spatial_ids::AbstractArray{<:CartesianIndex{2}}, 
    all_spatial_ids::AbstractArray{<:CartesianIndex{2}}=source_spatial_ids,
)
    # Get spatial indices (CartesianIndex) of valid targets that are also spatial indices of sources
    target_spatial_ids = _target_spatial_ids(target_quality_spatial, source_spatial_ids)
    # Find the node ids (Int) for the source row corresponding with spatial indices

    target_nodes = _find_all_sorted(source_spatial_ids, target_spatial_ids)
    target_graph_ids = _find_all_sorted(all_spatial_ids, target_spatial_ids)
    # Return Vector{NamedTuple} each with target.spatial and target.node
    return map(target_spatial_ids, target_graph_ids, eachindex(target_nodes), target_nodes) do spatialidx, gridgraphidx, connectedgraphidx, node
        (; spatialidx, gridgraphidx, connectedgraphidx, node)
    end
end

function _find_all_sorted(haystack, needles)
    i = 0
    ids = Vector{Int}(undef, length(needles))
    for (j, needle) in enumerate(needles)
        while i <= length(haystack)
            i += 1
            if haystack[i] == needle
                ids[j] = i
                break
            end
        end
    end
    return ids
end

"""
    is_strongly_connected(g::GridGraph)::Bool

Test if a GridGraph is fully connected.

# Examples

```jldoctests
julia> affinities = [1/4 0 1/4 1/4
                     1/4 0 1/4 1/4
                     1/4 0 1/4 1/4
                     1/4 0 1/4 1/4];

julia> grid = ConScape.Grid(size(affinities)..., affinities=ConScape.graph_matrix_from_raster(affinities), prune=false)
ConScape.Grid of size 4x4

julia> ConScape.is_strongly_connected(grid)
false
```
"""
function Graphs.is_strongly_connected(g::GridGraph)
    isnothing(transitionlikelihood(g)) && throw(ArgumentError("GridGraph has no transitionlikelihood"))
    Graphs.is_strongly_connected(SimpleWeightedDiGraph(transitionlikelihood(g)))
end
function Graphs.is_strongly_connected(g::ConnectedGraph)
    isnothing(transitionlikelihood(g)) && throw(ArgumentError("ConnectedGraph has no transitionlikelihood"))
    Graphs.is_strongly_connected(SimpleWeightedDiGraph(transitionlikelihood(g)))
end

# This neighborhood ordering makes i ordered for the sparse matrix
const N4 = (( 0, -1, 1.0, 1), # E
            (-1,  0, 1.0, 2), # S
            ( 1,  0, 1.0, 3), # N
            ( 0,  1, 1.0, 4)) # W

const N8 = ((-1, -1,  √2, 1), # SE
            ( 0, -1, 1.0, 4), # E
            ( 1, -1,  √2, 6), # NE
            (-1,  0, 1.0, 2), # S
            ( 1,  0, 1.0, 5), # W
            (-1,  1,  √2, 3), # SW
            ( 0,  1, 1.0, 7), # N
            ( 1,  1,  √2, 8)) # NW

# TODO document
abstract type AdjacencyWeight end

struct TargetWeight <: AdjacencyWeight end
struct AverageWeight <: AdjacencyWeight end

# TODO document these equations
weightedval(::TargetWeight, ::Likelihood, baseval, targetval, distance) =
    targetval / distance
weightedval(::TargetWeight, ::Cost, baseval, targetval, distance) =
    targetval * distance
weightedval(::AverageWeight, ::Cost, baseval, targetval, distance) =
    ((baseval + targetval) * distance) / 2
weightedval(::AverageWeight, ::Likelihood, baseval, targetval, distance) =
    2 / ((inv(baseval) + inv(targetval)) * distance)

"""
    graph_matrix_from_raster(R::Matrix; kw...) -> SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix of the raster image `R` 
of cell affinities or cell costs. The values are computed as either the value of 
the target cell (TargetWeight) or as harmonic (arithmetic) means of the cell 
affinities (costs) weighted by the grid distance (AverageWeight). 

The values can be computed with respect to eight `neighbors`` (`N8`) or four neighbors (`N4`).

# Keywords

- `transition_weight`: `TargetWeight` or `AverageWeight`, TargetWeight by default.
- `neighbors` : `N4` or `N8`, `N8` by default.
- `input_type`: `Likelyhood()` or `Cost()`, `Likelyhood()` by default
"""
function graph_matrix_from_raster(R::AbstractMatrix;
    neighbors::Tuple=N8,
    transition_weight=TargetWeight(),
    input_type,
)
    m, n = size(R)
    len = count(x -> !isnan(x) && !iszero(x), R) * 7
    # Initialize the buffers of the SparseMatrixCSC
    is, js, vals = Int[], Int[], Float64[]
    sizehint!(is, len)
    sizehint!(js, len)
    sizehint!(vals, len)

    for j in 1:n, i in 1:m
        # Base node
        baseval = R[i, j]
        for (ki, kj, distance, _) in neighbors
            # Continue when computing edge out of raster image
            (!(1 <= i + ki <= m) || !(1 <= j + kj <= n)) && continue
            # Target node
            targetval = R[i + ki, j + kj]
            (iszero(targetval) || isnan(targetval)) && continue
            val = weightedval(transition_weight, input_type, baseval, targetval, distance)
            # Add edge
            _maybe_push_sparse!(is, js, vals, m, n, i, j, ki, kj, val)
        end
    end
    return sparse(is, js, vals, m*n, m*n)
end
# A 3 dimensional Array already encodes edge weights
function graph_matrix_from_raster(R::AbstractArray{<:Any,3};
    neighbors::Tuple=N8,
    transition_weight=TargetWeight(),
    input_type,
)
    nneighbors, m, n = size(R)
    # Initialize the buffers of the SparseMatrixCSC
    is, js, vals = Int[], Int[], Float64[]

    for j in 1:n, i in 1:m
        if nneighbors == 4
            for (ki, kj, _, k) in N4
                val = R[k, i, j]
                _maybe_push_sparse!(is, js, vals, m, n, i, j, ki, kj, val)
            end
        elseif nneighbors == 8
            for (ki, kj, _, k) in N8
                val = R[k, i, j]
                _maybe_push_sparse!(is, js, vals, m, n, i, j, ki, kj, val)
            end
        else
            throw(ArgumentError("R must be a 3D array with the first dimension of length 4 or 8"))
        end
    end
    return sparse(is, js, vals, m*n, m*n)
end

@inline function _maybe_push_sparse!(is, js, vals, m, n, i, j, ki, kj, val)
    # Continue when computing edge out of raster image
    (!(1 <= i + ki <= m) || !(1 <= j + kj <= n)) && return nothing
    # Don't include zero or NaN similarities
    (iszero(val) || isnan(val)) && return nothing
    push!(is, (j - 1) * m + i)
    push!(js, (j - 1) * m + i + ki + kj*m)
    push!(vals, val)
    return nothing
end

"""
    graph_matrix_from_geometries(geoms::AbstractVector; kw...) -> SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix from the geometies `geoms`.

# Keywords

- `input_type`: `Likelyhood()` or `Cost()`.
- `transition_weight`: `TargetWeight()` or `AverageWeight()`, `TargetWeight()` by default.
- `cutoff_distance`: the distance at which to stop computing affinities.
    Should be in the same units as the geometry projection.
"""
function graph_matrix_from_geometries(geoms::AbstractVector, values::AbstractVector;
    transition_weight=TargetWeight(),
    cutoff_distance,
    input_type,
)
    ngeoms = length(geoms)
    # Make a tree of geometry extents for fast lookup
    tree = STR.STRtree(geoms)
    # Initialize the buffers of the SparseMatrixCSC
    is, js, vals = Int[], Int[], Float64[]

    # Lookup over all geometries
    for i in 1:ngeoms
        geom = geoms[i]
        baseval = values[i]
        center = GO.centroid(geom)
        # Make a square region around the geometry
        region = Extents.buffer(GI.extent(geom), (X=cutoff_distance, Y=cutoff_distance))
        # Find all neighboring geoms inside the region
        neighbor_inds = STR.query(tree, region)
        # Iterate over indices and distances, adding neighbors closer than `cutoff_distance`
        for j in neighbor_inds
            # If the geom is too far away, skip it
            distance = GO.distance(center, geoms[j]) 
            distance > cutoff_distance && continue
            # Get the distance weigth value
            targetval = values[j]
            val = weightedval(transition_weight, input_type, baseval, targetval, distance)
            # Add edge for this geometry
            # i is the current geometry index
            push!(is, i)
            # j is the found neighbor index
            push!(js, j)
            # val is the weighted affinity or cost
            push!(vals, val)
        end
    end
    # Generate a sparse array from i and j indices and values
    return sparse(is, js, vals, ngeoms, ngeoms)
end
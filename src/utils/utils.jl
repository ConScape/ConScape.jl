
"""
    mapnz(f, A::SparseMatrixCSC)::SparseMatrixCSC

Map the non-zero values of a sparse matrix `A` with the function `f`.
"""
function mapnz(f, A::SparseMatrixCSC)
    B = copy(A)
    map!(f, B.nzval, A.nzval)
    return B
end
function mapnz(f, A::AbstractArray)
    B = copy(A)
    map!(f, B.data, A.data)
    return B
end

_prepare_qualities(A::AbstractMatrix) = _no_nan_f64.(_unwrap_raster(A))
_no_nan_f64(x) = Float64(x) # == isnan(x) ? 0.0 : Float64(x)
_unwrap_raster(R::Raster) = parent(R)
_unwrap_raster(R::AbstractMatrix) = R

function _maybe_raster(
    measures::Union{MeasureTuple,MeasureNamedTuple}, 
    mats::Union{Tuple,NamedTuple}, 
    g::Initialisation
)
    map(measures, mats) do measure, mat 
        _maybe_raster(returntrait(measure), mat, dims(g); name=Symbol(measure))
    end
end
_maybe_raster(rt, mat::Raster, g::Initialisation; kw...) = mat
_maybe_raster(rt, x, y; kw...) = x
_maybe_raster(rt, mat::AbstractMatrix, g::Initialisation; kw...) =
    _maybe_raster(rt, mat, dims(g); kw...)
_maybe_raster(rt::DenseSpatial, mat::Matrix{T}, dims::Tuple; kw...) where T =
    Raster(mat, dims; missingval=T(NaN), kw...)
_maybe_raster(rt, vec::Vector{T}, dims::Tuple; kw...) where T =
    Raster(vec, dims; missingval=T(NaN), kw...)

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
    return map(target_spatial_ids, target_graph_ids, eachindex(target_nodes), target_nodes) do spatialidx, graphidx, subgraphidx, node
        (; spatialidx, graphidx, subgraphidx, node)
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

function _fill_matrix(values, g::Initialisation)
    matrix = fill(NaN, size(g))
    matrix[source_ids(g)] .= values
    return matrix
end

function Raster(values::AbstractVector, p::Initialisation; kw...)
    ds = dims(p)
    isnothing(ds) && throw(ArgumentError("dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, p), ds::Tuple; kw...)
end

# function outdegrees(p::Initialisation)
#     values = sum(affinitymatrix(p), dims=2)
#     _maybe_raster(_fill_matrix(values, p), p)
# end

# function indegrees(p::Initialisation; kwargs...)
#     g = grid(p)
#     values = sum(affinitymatrix(g), dims=1)
#     _maybe_raster(_fill_matrix(values, p), p)
# end

"""
    is_strongly_connected(g::GridGraph)::Bool

Test if graph defined by Grid is fully connected.

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
Graphs.is_strongly_connected(g::GridGraph) = 
    Graphs.is_strongly_connected(SimpleWeightedDiGraph(g.transitionlikelihood))

function split_subgraphs(g::GridGraph;
    costfunction=nothing, likelihoodfunction=nothing,
)
    spatialidxs = vec(CartesianIndices(size(g)))
    targetids = _target_ids(targetquality(g), spatialidxs)
    # Convert cost matrix to graph, todo: is `permute=false` needed
    graph = SimpleWeightedDiGraph(
        isnothing(transitioncost(g)) ? transitionlikelihood(g) : transitioncost(g); 
        permute=false
    )

    # Find the subgraphs
    scc = Graphs.strongly_connected_components(graph)

    # Keep all subgraphs that contain target nodes
    subgraphs_with_targets = map(scc) do c
        length(c) > 1 && any(t -> t.node in c, targetids)
    end

    # Sort subgraphs by number of nodes
    subgraphs = sort!(scc[subgraphs_with_targets]; by=length, rev=true)

    # Return a Vector of Grids for each subgraph
    return map(subgraphs) do scci
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

        # Return new grid for subgraph
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

_maybe_set_diagonal!(ti::TargetInit, proximities) =
    _maybe_set_diagonal!(ti, proximities, diagvalue(ti))
_maybe_set_diagonal!(ti::TargetInit, proximities, diagvalue::Nothing) = proximities
function _maybe_set_diagonal!(ti::TargetInit, proximities, diagvalue::Number)
    proximities = ti.workspace .= proximities
    proximities[target(ti).node] = diagvalue
    return ReadOnlyArray(proximities)
end
# function _maybe_set_diagonal!(proximitymatrix, diagvalue::Number, targetnodes::AbstractVector)
# , diagvalue(ti), target(ti).node
#     for (j, i) in enumerate(targetnodes)
#         proximitymatrix[i, j] = diagvalue
#     end
# end

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
        v = vec(A)::Vector
        resize!(v, len)
        reshape(v, size)
    end
end

_allocate_workspaces!(x, problem::Problem, graph::ConnectedGraph) =
    _allocate_workspaces!(x, problem, nsources(graph))
_allocate_workspaces!(x::Nothing, problem::Problem, length::Int) =
    Workspaces(length, nworkspaces(problem) + 20)
_allocate_workspaces!(workspaces::Workspaces, ::Problem, length::Int) =
    free!(resize!(workspaces, length))

_maybe_new_outputs(mes, mgi) =
    mes === measures(mgi) ? outputs(mgi) : allocate_output(mes, mgi)

function _maybe_raster_outputs(measures, outputs, mgi)
    out = _maybe_raster(measures, outputs, mgi)
    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end

# Get layers from a RasterStack or return nothing
_get_sourcequality(rast::RasterStack) = _keys_or_nothing(rast, (:sourcequality, :quality))
_get_targetquality(rast::RasterStack) = _keys_or_nothing(rast, (:targetquality, :quality, :sourcequality))
_get_likelihood(rast::RasterStack) = _keys_or_nothing(rast, (:likelihood, :movementlikelihood))
_get_cost(rast::RasterStack) = _keys_or_nothing(rast, (:cost, :movementcost))

@inline _keys_or_nothing(rast, (key, keys...)::Tuple) =
    haskey(rast, key) ? rast[key] : _keys_or_nothing(rast, keys)
@inline _keys_or_nothing(rast, ::Tuple{}) = nothing
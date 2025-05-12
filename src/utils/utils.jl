
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

_maybe_raster(measures::Union{MeasureTuple,MeasureNamedTuple}, mats::Union{Tuple,NamedTuple}, g::Initialisation) =
    map((measure, mat) -> _maybe_raster(returntrait(measure), mat, dims(g)), measures, mats)
_maybe_raster(rt, mat::Raster, g::Initialisation) = mat
_maybe_raster(rt, mat::AbstractMatrix, g::Initialisation) =
    _maybe_raster(rt, mat, dims(g))
_maybe_raster(rt, x, _) = x
_maybe_raster(rt::DenseSpatial, mat::Matrix{T}, dims::Tuple) where T =
    Raster(mat, dims; missingval=T(NaN))
_maybe_raster(rt, vec::Vector{T}, dims::Tuple) where T =
    Raster(vec, dims; missingval=T(NaN))

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

function _target_ids(target_quality_spatial::AbstractMatrix, source_spatial_ids::Vector{CartesianIndex{2}}, all_spatial_ids::Vector{CartesianIndex{2}})
    # Get spatial indices (CartesianIndex) of valid targets that are also spatial indices of sources
    target_spatial_ids = _target_spatial_ids(target_quality_spatial, source_spatial_ids)
    # Find the node ids (Int) for the source row corresponding with spatial indices
    target_nodes = findall(source_spatial_ids) do id
        id in target_spatial_ids
    end
    target_grid_ids = findall(all_spatial_ids) do id
        id in target_spatial_ids
    end
    # Return Vector{NamedTuple} each with target.spatial and target.node
    return map(target_spatial_ids, target_grid_ids, eachindex(target_nodes), target_nodes) do spatial, grid_id, subgrid_id, node
        (; spatial, grid_id, subgrid_id, node)
    end
end

function _fill_matrix(values, g::Initialisation)
    matrix = fill(NaN, size(g))
    matrix[source_ids(g)] .= values
    return matrix
end

function Raster(values::AbstractVector, p::Initialisation; kw...)
    ds = dims(p)
    isnothing(ds) && throw(ArgumentError("Grid dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, p), ds::Tuple; kw...)
end

function outdegrees(p::Initialisation)
    values = sum(affinitymatrix(p), dims=2)
    _maybe_raster(_fill_matrix(values, p), p)
end

function indegrees(p::Initialisation; kwargs...)
    g = grid(p)
    values = sum(affinitymatrix(g), dims=1)
    _maybe_raster(_fill_matrix(values, p), p)
end

"""
    is_strongly_connected(g::Grid)::Bool

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
Graphs.is_strongly_connected(g::Grid) = is_strongly_connected(SimpleWeightedDiGraph(g.affinitymatrix))

function split_subgraphs(g::Grid)
    # Convert cost matrix to graph, todo: is `permute=false` needed
    graph = SimpleWeightedDiGraph(costmatrix(g); permute=false)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    # Keep all subgraphs that contain target nodes
    subgraphs_with_targets = map(scc) do c
        length(c) > 1 && any(t -> t.node in c, g.target_ids)
    end

    # Sort subgraphs by number of nodes
    subgraphs = sort!(scc[subgraphs_with_targets]; by=length, rev=true)

    # Return a Vector of Grids for each subgraph
    return map(subgraphs) do scci
        # Sort subgraph indices
        sort!(scci)

        # Get matrices for the subgraph 
        affinitymatrix = g.affinitymatrix[scci, scci]
        costmatrix = isnothing(g.costfunction) ? g.costmatrix[scci, scci] : mapnz(g.costfunction, affinitymatrix)

        # Get new source and target ids for subgraph
        source_ids = g.source_ids[scci]
        all_spatial_ids = vec(collect(CartesianIndices(size(g))))
        target_ids = _target_ids(g.target_quality_spatial, source_ids, all_spatial_ids)

        # Get source and target quality vectors for subgraph
        source_quality_vector = [g.source_quality_spatial[i] for i in source_ids]
        target_quality_vector = [g.target_quality_spatial[i.spatial] for i in target_ids]

        # Return new grid for subgraph
        Grid(
            g.size,
            g.costfunction,
            costmatrix,
            affinitymatrix,
            g.source_quality_spatial, g.target_quality_spatial,
            source_quality_vector, target_quality_vector,
            source_ids, target_ids,
            g.dims,
        )
    end
end
function _get_target_qualities(rast::AbstractRasterStack)
    get(rast, :target_qualities) do
        get(rast, :qualities) do
            throw(ArgumentError("No :target_qualities or :qualities layers found"))
        end
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

_allocate_workspaces!(x, problem::Problem, grid::Grid) =
    _allocate_workspaces!(x, problem, nsources(grid))
_allocate_workspaces!(x::Nothing, problem::Problem, length::Int) =
    Workspaces(length, nworkspaces(problem) + 20)
_allocate_workspaces!(workspaces::Workspaces, ::Problem, length::Int) =
    free!(resize!(workspaces, length))

_maybe_new_outputs(mes, mgi) =
    mes === measures(mgi) ? outputs(mgi) : allocate_output(mes, mgi)
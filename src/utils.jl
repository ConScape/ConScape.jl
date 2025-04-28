const N4 = (( 0, -1, 1.0),
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            ( 0,  1, 1.0))
const N8 = ((-1, -1,  √2),
            ( 0, -1, 1.0),
            ( 1, -1,  √2),
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            (-1,  1,  √2),
            ( 0,  1, 1.0),
            ( 1,  1,  √2))

# TODO document
abstract type AdjacencyWeight end

struct TargetWeight <: AdjacencyWeight end
struct AverageWeight <: AdjacencyWeight end

abstract type MatrixType end

struct AffinityMatrix <: MatrixType end
struct CostMatrix <: MatrixType end

# TODO document these equations
weightedval(::TargetWeight, ::AffinityMatrix, baseval, targetval, distance) =
    targetval / distance
weightedval(::TargetWeight, ::CostMatrix, baseval, targetval, distance) =
    targetval * distance
weightedval(::AverageWeight, ::CostMatrix, baseval, targetval, distance) =
    2 / ((inv(baseval) + inv(targetval)) * distance)
weightedval(::AverageWeight, ::AffinityMatrix, baseval, targetval, distance) =
    ((baseval + targetval) * distance) / 2

"""
    graph_matrix_from_raster(R::Matrix; kw...) -> SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix of the raster image `R` 
of cell affinities or cell costs. The values are computed as either the value of 
the target cell (TargetWeight) or as harmonic (arithmetic) means of the cell 
affinities (costs) weighted by the grid distance (AverageWeight). 

The values can be computed with respect to eight `neighbors`` (`N8`) or four neighbors (`N4`).

# Keywords

- `matrix_type`: `AffinityMatrix` or `CostMatrix`, `AffinityMatrix` by default.
- `weight`: `TargetWeight` or `AverageWeight`, TargetWeight by default.
- `neighbors` : `N4` or `N8`, `N8` by default.
"""
function graph_matrix_from_raster(R::AbstractMatrix;
    neighbors::Tuple=N8,
    matrix_type=AffinityMatrix(),
    weight=TargetWeight(),
)
    m, n = size(R)
    # Initialize the buffers of the SparseMatrixCSC
    is, js, vals = Int[], Int[], Float64[]

    for j in 1:n, i in 1:m
        # Base node
        baseval = R[i, j]
        for (ki, kj, distance) in neighbors
            # Continue when computing edge out of raster image
            (!(1 <= i + ki <= m) || !(1 <= j + kj <= n)) && continue
            # Target node
            targetval = R[i + ki, j + kj]
            # Don't include zero or NaN similaritiers
            iszero(targetval) || isnan(targetval) && continue
            # Add edge
            val = weightedval(weight, matrix_type, baseval, targetval, distance)
            push!(is, (j - 1)*m + i)
            push!(js, (j - 1)*m + i + ki + kj*m)
            push!(vals, val)
        end
    end
    return sparse(is, js, vals, m*n, m*n)
end

"""
    graph_matrix_from_geometries(geoms::AbstractVector; kw...) -> SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix from the geometies `geoms`.

# Keywords

- `matrix_type`: `AffinityMatrix()` or `CostMatrix()`, `AffinityMatrix()` by default.
- `weight`: `TargetWeight()` or `AverageWeight()`, `TargetWeight()` by default.
- `cutoff_distance`: the distance at which to stop computing affinities.
    Should be in the same units as the geometry projection.
"""
function graph_matrix_from_geometries(geoms::AbstractVector, values::AbstractVector;
    matrix_type=AffinityMatrix(),
    weight=TargetWeight(),
    cutoff_distance,
)
    ngeoms = length(geoms)
    # Make a tree of geometry extents for fast lookup
    tree = STRtree(geoms)
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
        neighbor_inds = query(tree, region)
        # Iterate over indices and distances, adding neighbors closer than `cutoff_distance`
        for j in neighbor_inds
            # If the geom is too far away, skip it
            distance = GO.distance(center, geoms[j]) 
            distance > cutoff_distance && continue
            # Get the distance weigth value
            targetval = values[j]
            val = weightedval(weight, matrix_type, baseval, targetval, distance)
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

#=
Make pixels impossible to move to by changing the affinities to them to zero.
Input:
    - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
=#
function _set_impossible_nodes(g::Grid, node_list::Vector{CartesianIndex{2}}, impossible_affinity=1e-20)
    # Find the indices of the coordinates in the source_ids vector
    node_list_idx = [findfirst(isequal(n), source_ids(g)) for n in node_list]

    # Copy affinities and qualities for modification
    affinitymatrix = copy(g.affinitymatrix)
    source_qualities = copy(g.source_quality_spatial)
    target_qualities = copy(g.target_quality_spatial)

    # Set (nonzero) values to impossible_affinity:
    # affinitymatrix
    # FIXME! Row slicing of a sparse matrix is really inefficient
    affinitymatrix[node_list_idx, :] = impossible_affinity * (affinitymatrix[node_list_idx, :] .> 0)
    affinitymatrix[:, node_list_idx] = impossible_affinity * (affinitymatrix[:, node_list_idx] .> 0)

    dropzeros!(affinitymatrix)

    # Qualities
    source_qualities[node_list] .= 0
    target_qualities[node_list] .= 0

    # Generate a new Grid based on the modified affinitymatrix
    return Grid(size(g); affinitymatrix, source_qualities, target_qualities, costfunction=g.costfunction, costmatrix=g.costmatrix)
end

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

_maybe_raster(mat::Raster, g::Initialisation) = mat
_maybe_raster(mat::AbstractMatrix, g::Initialisation) =
    _maybe_raster(mat, dims(g))
_maybe_raster(mats::Union{Tuple,NamedTuple}, g::Initialisation) =
    map(mat -> _maybe_raster(mat, g), mats)
_maybe_raster(x, _) = x
_maybe_raster(mat::Matrix{T}, dims::Tuple) where T =
    Raster(mat, dims; missingval=T(NaN))

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

function Raster(values::AbstractVector, p::Initialisation; kwargs...)
    isnothing(dims(p)) && throw(ArgumentError("Grid dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, p), dims(p); kwargs...)
end

function outdegrees(p::Initialisation)
    values = sum(affinitymatrix(p), dims=2)
    _maybe_raster(_fill_matrix(values, p), p)
end

function indegrees(p::Initialisation; kwargs...)
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
        costmatrix = g.costfunction === nothing ? g.costmatrix[scci, scci] : mapnz(g.costfunction, affinitymatrix)

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

"""
    sum_neighborhood(g::Grid, rc::Tuple{Int,Int}, npix::Integer)::Float64

A helper-function, used by coarse_graining, that computes the sum of pixels within a npix neighborhood around the target rc.
"""
sum_neighborhood(g, rc, npix) = sum_neighborhood(g.target_qualities, rc, npix)
function sum_neighborhood(target_qualities::AbstractMatrix, rc, npix)
    getrows = (rc[1]-floor(Int, npix / 2)):(rc[1]+(ceil(Int, npix / 2)-1))
    getcols = (rc[2]-floor(Int, npix / 2)):(rc[2]+(ceil(Int, npix / 2)-1))
    # pixels outside of the landscape are encoded with NaNs but we don't want
    # the NaNs to propagate to the coarse grained values
    return sum(t -> isnan(t) ? 0.0 : t, target_qualities[getrows, getcols])
end

"""
    coarse_graining(g::Grid, npix::Integer)::Array

Creates a sparse matrix of target qualities for the landmarks based on merging npix pixels into the center pixel.
"""
function coarse_graining(g, npix)
    coarse_graining(g.target_quality_spatial, npix;
        source_ids=source_ids(g)
    )
end
coarse_graining(rast::AbstractRaster, npix; kw...) =
    rebuild(rast, coarse_graining(parent(rast), npix; kw...))
function coarse_graining(rast::AbstractRasterStack, npix; kw...)
    target = _get_target_qualities(rast)
    # Get target qualities or qualities
    target_qualities = coarse_graining(target, npix; kw...)
    return Base.setindex(rast, target_qualities, :target_qualities)
end
function coarse_graining(M::AbstractMatrix, npix;
    source_ids=_id_gc_list(size(M)...)
)
    nrows, ncols = size(M)
    getrows = (floor(Int, npix / 2)+1):npix:(nrows-ceil(Int, npix / 2)+1)
    getcols = (floor(Int, npix / 2)+1):npix:(ncols-ceil(Int, npix / 2)+1)
    coarse_target_rc = Base.product(getrows, getcols)
    coarse_target_ids = vec(
        [
        findfirst(
            isequal(CartesianIndex(ij)),
            source_ids
        ) for ij in coarse_target_rc
    ]
    )
    coarse_target_rc = [ij for ij in coarse_target_rc if !ismissing(ij)]
    filter!(!ismissing, coarse_target_ids)
    V = [sum_neighborhood(M, ij, npix) for ij in coarse_target_rc]
    I = first.(coarse_target_rc)
    J = last.(coarse_target_rc)
    target_mat = sparse(I, J, V, nrows, ncols)
    target_mat = dropzeros(target_mat)

    return target_mat
end

function _get_target_qualities(rast::AbstractRasterStack)
    get(rast, :target_qualities) do
        get(rast, :qualities) do
            throw(ArgumentError("No :target_qualities or :qualities layers found"))
        end
    end
end

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

# _newstoragedict(::Workspaces{W}) where W = Dict{Symbol,W}()
_newstoragedict(::Workspaces{W}) where W = Dict{Symbol,Any}()
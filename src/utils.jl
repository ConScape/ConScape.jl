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

@enum AdjacencyWeight begin
    TargetWeight
    AverageWeight
end

@enum MatrixType begin
    AffinityMatrix
    CostMatrix
end

"""
    graph_matrix_from_raster(R::Matrix[, type=AffinityMatrix, neighbors::Tuple=N8, weight=TargetWeight])::SparseMatrixCSC

Compute a graph matrix, i.e. an affinity or cost matrix of the raster image `R` of cell affinities or cell costs.
The values are computed as either the value of the target cell (TargetWeight) or as harmonic (arithmetic) means
of the cell affinities (costs) weighted by the grid distance (AverageWeight). The values can be computed with
respect to eight neighbors (`N8`) or four neighbors (`N4`).
"""
function graph_matrix_from_raster(
    R::AbstractMatrix;
    matrix_type=AffinityMatrix,
    neighbors::Tuple=N8,
    weight=TargetWeight
)
    m, n = size(R)

    # Initialize the buffers of the SparseMatrixCSC
    is, js, vs = Int[], Int[], Float64[]

    for j in 1:n
        for i in 1:m
            # Base node
            rij = R[i, j]
            for (ki, kj, l) in neighbors
                if !(1 <= i + ki <= m) || !(1 <= j + kj <= n)
                    # Continue when computing edge out of raster image
                    continue
                else
                    # Target node
                    rijk = R[i + ki, j + kj]
                    if iszero(rijk) || isnan(rijk)
                        # Don't include zero or NaN similaritiers
                        continue
                    end

                    push!(is, (j - 1)*m + i)
                    push!(js, (j - 1)*m + i + ki + kj*m)
                    if weight == TargetWeight
                        if matrix_type == AffinityMatrix
                            push!(vs, rijk/l)
                        elseif matrix_type == CostMatrix
                            push!(vs, rijk*l)
                        end
                    elseif weight == AverageWeight
                        if matrix_type == AffinityMatrix
                            v = 2/((inv(rij) + inv(rijk))*l)
                            push!(vs, v)
                        elseif matrix_type == CostMatrix
                            v = ((rij + rijk)*l)/2
                            push!(vs, v)
                        end
                    else
                        throw(ArgumentError("weight mode $weight not implemented"))
                    end
                end
            end
        end
    end
    # TODO just make a BandedMatrix from the start
    # and what happens when this is not square?
    return (sparse(is, js, vs, m*n, m*n))
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

_maybe_raster(mat::Raster, g::Initialisation) = mat
_maybe_raster(mat::AbstractMatrix, g::Initialisation) =
    _maybe_raster(mat, dims(g))
_maybe_raster(mats::Union{Tuple,NamedTuple}, g::Initialisation) =
    map(mat -> _maybe_raster(mat, g), mats)
_maybe_raster(x, _) = x
_maybe_raster(mat::Matrix{T}, dims::Tuple) where T =
    Raster(mat, dims; missingval=T(NaN))

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
        target_ids = _target_ids(g.target_quality_spatial, source_ids)

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
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
    return Grid(size(g); affinitymatrix, source_qualities, target_qualities, g.costfunction, g.costmatrix)
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

# Helper to get keyword arguments
function _keywords(o::T) where T
    vals = map(f -> getfield(o, f), fieldnames(T))
    return NamedTuple{fieldnames(T)}(vals) 
end

_maybe_raster(mat::Raster, g::Initialisation) = mat
_maybe_raster(mat::AbstractMatrix, g::Initialisation) =
    _maybe_raster(mat, dims(g))
_maybe_raster(mats::Union{Tuple,NamedTuple}, g::Initialisation) =
    map(mat -> _maybe_raster(mat, g), mats)
_maybe_raster(x, _) = x
_maybe_raster(mat::Matrix{T}, dims::Tuple) where T =
    Raster(mat, dims; missingval=T(NaN))
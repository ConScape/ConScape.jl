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

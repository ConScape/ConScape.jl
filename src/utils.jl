# simulate a permeable wall
function perm_wall_sim(
    nrows::Integer,
    ncols::Integer;
    scaling::Float64=0.5,
    wallwidth::Integer=3,
    wallposition::Float64=0.5,
    corridorwidths::NTuple{<:Any,Int}=(3,3),
    corridorpositions=(0.35,0.7),
    impossible_affinity::Real=1e-20,
    nhood_size::Integer=8,
    kwargs...
)

    # 1. initialize landscape
    affinities = _generate_affinities(nrows, ncols, nhood_size)
    g = Grid(nrows, ncols; affinities=affinities*scaling, kwargs...)

    # # 2. compute the wall
    wpt = round(Int, ncols*wallposition - wallwidth/2 + 1)
    xs  = range(wpt, stop=wpt + wallwidth - 1)

    # 3. compute the corridors
    ys = Int[]
    for i in 1:length(corridorwidths)
        cpt = floor(Int, nrows*corridorpositions[i]) - ceil(Int, corridorwidths[i]/2)
        if i == 1
            append!(ys, 1:cpt)
        else
            append!(ys, range(maximum(ys) + 1 + corridorwidths[i-1], stop=cpt))
        end
    end
    append!(ys, range(maximum(ys) + 1 + corridorwidths[end]  , stop=nrows))

    impossible_nodes = vec(CartesianIndex.(collect(Iterators.product(ys, xs))))
    g = _set_impossible_nodes(g, impossible_nodes, impossible_affinity)

    return g
end

⊗ = kron
#=
Generate the affinity matrix of a grid graph, where each
pixel is connected to its vertical and horizontal neighbors.

Parameters:
- nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
=#
function _generate_affinities(nrows, ncols, nhood_size)

    if !(nhood_size ∈ (4, 8))
        throw(ArgumentError("nhood_size must be either 4 or 8"))
    end

    affinities = spdiagm(0=>ones(ncols)) ⊗ spdiagm(-1=>ones(nrows - 1), 1=>ones(nrows - 1)) +
        spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗ spdiagm(0=>ones(nrows))

    if nhood_size == 8
        affinities .+= spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗
              spdiagm(-1=>fill(1/√2, nrows - 1), 1=>fill(1/√2, nrows - 1))
    end

    return affinities
end

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
weightedval(::TargetWeight, ::CostMatrix, baseval, targetval, distance) =
    targetval / distance
weightedval(::TargetWeight, ::AffinityMatrix, baseval, targetval, distance) =
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
    # Find the indices of the coordinates in the id_to_grid_coordinate_list vector
    node_list_idx = [findfirst(isequal(n), g.id_to_grid_coordinate_list) for n in node_list]

    # Copy affinities and qualities for modification
    affinities       = copy(g.affinities)
    source_qualities = copy(g.source_qualities)
    target_qualities = copy(g.target_qualities)

    # Set (nonzero) values to impossible_affinity:
    # Affinities
    # FIXME! Row slicing of a sparse matrix is really inefficient
    affinities[node_list_idx,:] = impossible_affinity*(affinities[node_list_idx,:] .> 0)
    affinities[:,node_list_idx] = impossible_affinity*(affinities[:,node_list_idx] .> 0)
    dropzeros!(affinities)

    # Qualities
    source_qualities[node_list] .= 0
    target_qualities[node_list] .= 0

    # Generate a new Grid based on the modified affinities
    costs = g.costfunction === nothing ? g.costmatrix : g.costfunction
    return Grid(size(g)...; affinities, source_qualities, target_qualities, costs)
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
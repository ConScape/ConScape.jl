
const TargetID = @NamedTuple{spatial::CartesianIndex{2},node::Int}
const SourceID = CartesianIndex{2}

abstract type Precalculations end

costfunction(p::Precalculations) = costfunction(grid(p))
costmatrix(p::Precalculations) = costmatrix(grid(p))
affinitymatrix(p::Precalculations) = affinitymatrix(grid(p))
source_quality_vector(p::Precalculations) = source_quality_vector(grid(p))
target_quality_vector(p::Precalculations) = target_quality_vector(grid(p))
source_quality_spatial(p::Precalculations) = source_quality_spatial(grid(p))
target_quality_spatial(p::Precalculations) = target_quality_spatial(grid(p))
source_ids(p::Precalculations) = source_ids(grid(p))
target_ids(p::Precalculations) = target_ids(grid(p))
movement_mode(p::Precalculations) = movement_mode(problem(p))
solver(p::Precalculations) = solver(problem(p))
graph_measures(p::Precalculations) = graph_measures(problem(p))
connectivity_measure(p::Precalculations) = connectivity_measure(problem(p))

nsources(p::Precalculations) = length(source_ids(p))
ntargets(p::Precalculations) = length(target_ids(p))
sparse_size(p::Precalculations) = nsources(p), ntargets(p) 

Base.size(p::Precalculations) = Base.size(grid(p))
DimensionalData.dims(p::Precalculations) = DimendalData.dims(grid(p))

"""
    Grid(nrows::Integer,
         ncols::Integer;
         affinitymatrix=nothing,
         qualities::Matrix=ones(nrows, ncols),
         source_qualities::Matrix=qualities,
         target_qualities::AbstractMatrix=qualities,
         costs::Union{Transformation,SparseMatrixCSC{Float64,Int}}=MinusLog(),
         prune=true)::Grid

Construct a `Grid` from an `affinitymatrix` matrix of type `SparseMatrixCSC`. 

It is possible to also supply matrices of `source_qualities` and `target_qualities` as well as
a `costs` function that maps the `affinitymatrix` matrix to a `costs` matrix. 

Alternatively, it is possible to supply a matrix to `costs` directly. If `prune=true` (the default), 
the affinity and cost matrices will be pruned to exclude unreachable nodes.
"""
struct Grid{D<:Union{Tuple,Nothing},F<:Union{Nothing,Transformation},SQ,TQ} <: Precalculations
    size::Tuple{Int,Int}
    costfunction::F
    costmatrix::SparseMatrixCSC{Float64,Int}
    affinitymatrix::SparseMatrixCSC{Float64,Int}
    source_quality_spatial::SQ
    target_quality_spatial::TQ
    source_quality_vector::Vector{Float64}
    target_quality_vector::Vector{Float64}
    source_ids::Vector{SourceID}
    target_ids::Vector{TargetID}
    dims::D
end
function Grid(size::Tuple{Int,Int};
    affinitymatrix::SparseMatrixCSC{Float64,Int},
    source_qualities::AbstractMatrix,
    target_qualities::AbstractMatrix=source_qualities,
    costfunction::Transformation=MinusLog(),
    check=false,
)
    if prod(size) != LinearAlgebra.checksquare(affinitymatrix)
        n = size(affinitymatrix, 1)
        throw(ArgumentError("grid size $size is incompatible with size of affinity matrix ($n, $n)"))
    end
    costmatrix = mapnz(costfunction, affinitymatrix)

    # This is too expensive to calculate for small target grids
    if check
        if any(t -> t < 0, nonzeros(costmatrix))
            throw(ArgumentError("The cost graph can have only non-negative edge weights. Perhaps you should change the cost function?"))
        end
        cost_digraph = SimpleDiGraph(costmatrix)
        affinity_digraph = SimpleDiGraph(affinitymatrix)

        if ne(difference(cost_digraph, affinity_digraph)) > 0
            throw(ArgumentError("cost graph contains edges not present in the affinity graph"))
        end
    end

    source_quality_spatial = _prepare_qualities(source_qualities)
    target_quality_spatial = _prepare_qualities(target_qualities)

    # Initially just every node
    source_ids = vec(collect(CartesianIndices(size)))
    # Subset of source_ids with valid quality
    target_ids = _target_ids(target_qualities, source_ids)
    # Initially just all spatial source qualities
    source_quality_vector = vec(source_quality_spatial)
    # Subset of spatial target qualities with valid quality
    target_quality_vector = [target_quality_spatial[t.spatial] for t in target_ids]

    g = Grid(
        size,
        costfunction,
        costmatrix,
        affinitymatrix,
        source_quality_spatial, target_quality_spatial,
        source_quality_vector, target_quality_vector,
        source_ids, target_ids,
        dims(source_qualities),
    )
    return g
end
function Grid(rast::RasterStack;
    affinitymatrix=ConScape.graph_matrix_from_raster(rast.affinities),
    source_qualities=rast.source_qualities,
    target_qualities=get(rast, :target_qualities, source_qualities),
    kw...
)
    Grid(size(rast); affinitymatrix, source_qualities, target_qualities, kw...)
end
Grid(p::AbstractProblem, rast::RasterStack; kw...) =
    Grid(rast; costfunction=costfunction(p), kw...)

affinitymatrix(g::Grid) = g.affinitymatrix
costmatrix(g::Grid) = g.costmatrix
source_quality_spatial(g::Grid) = g.source_quality_spatial
target_quality_spatial(g::Grid) = g.target_quality_spatial
source_quality_vector(g::Grid) = g.source_quality_vector
target_quality_vector(g::Grid) = g.target_quality_vector
source_ids(g::Grid) = g.source_ids
target_ids(g::Grid) = g.target_ids

Base.size(g::Grid) = g.size
function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    print(io, summary(g), " of size ", g.nrows, "x", g.ncols)
end

DimensionalData.dims(g::Grid) = g.dims

_prepare_qualities(A::AbstractMatrix) = _no_nan_f64.(_unwrap_raster(A))
_no_nan_f64(x) = isnan(x) ? 0.0 : Float64(x)
_unwrap_raster(R::Raster) = parent(R)
_unwrap_raster(R::AbstractMatrix) = R

# Compute a vector of the cartesian indices of nonzero target qualities and
# the corresponding node id corresponding to the indices
_target_spatial_ids(target_qauality::AbstractMatrix, source_spatial_ids::AbstractVector) = 
    source_spatial_ids
_target_spatial_ids(target_quality::Raster, source_spatial_ids::AbstractVector) = 
    _target_spatial_ids(parent(target_quality), source_spatial_ids)
function _target_spatial_ids(target_quality::SparseMatrixCSC, source_spatial_ids::AbstractVector)
    is, js, _ = findnz(target_quality)
    return intersect(CartesianIndex.(is, js), source_spatial_ids)
end

function _target_ids(target_quality_spatial::AbstractMatrix, source_spatial_ids::Vector{CartesianIndex{2}})
    # Get spatial indices (CartesianIndex) of valid targets that are also spatial indices of sources
    target_spatial_ids = _target_spatial_ids(target_quality_spatial, source_spatial_ids)
    # Find the node ids (Int) for the source row corresponding with spatial indices
    target_nodes = findall(source_spatial_ids) do id
        id in target_spatial_ids
    end
    println()
    @show length(source_spatial_ids) target_nodes 
    # Return Vector{NamedTuple} each with target.spatial and target.node
    return map(target_spatial_ids, target_nodes) do spatial, node
        (; spatial, node)
    end
end

function _fill_matrix(values, g::Precalculations)
    matrix = fill(NaN, size(g))
    matrix[source_ids(g)] .= values
    return matrix
end

function Raster(values::AbstractVector, p::Precalculations; kwargs...)
    isnothing(dims(p)) && throw(ArgumentError("Grid dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, p), dims(p); kwargs...)
end

function outdegrees(p::Precalculations)
    values = sum(affinitymatrix(p), dims=2)
    _maybe_raster(_fill_matrix(values, p), p)
end

function indegrees(p::Precalculations; kwargs...)
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
    largest_subgraph(g::Grid)::Grid

Extract the largest fully connected subgraph of the `Grid`. The returned `Grid`
will have the same size as the input `Grid` but only nodes associated with the
largest subgraph of the affinities will be active.
"""
function largest_subgraph(g::Grid)
    # Convert cost matrix to graph, todo: is `permute=false` needed
    graph = SimpleWeightedDiGraph(g.costmatrix, permute=false)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    # @info "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    _, i = findmax(length, scc)

    # extract node list and sort it
    scci = sort(scc[i])

    # ndiffnodes = size(g.costmatrix, 1) - length(scci)
    # if ndiffnodes > 0
    # @info "removing $ndiffnodes nodes from affinity and cost graphs"
    # end

    # Extract the adjacency matrix of the largest subgraph
    affinitymatrix = g.affinitymatrix[scci, scci]
    # affinities = convert(SparseMatrixCSC{Float64,Int}, graph[scci])

    costmatrix = g.costfunction === nothing ? g.costmatrix[scci, scci] : mapnz(g.costfunction, affinitymatrix)
    id_to_grid_coordinate_list = g.id_to_grid_coordinate_list[scci]
    targetidx, targetnodes = _targetidx_and_nodes(g.target_qualities, id_to_grid_coordinate_list)
    qs = [g.source_qualities[i] for i in id_to_grid_coordinate_list]
    qt = [g.target_qualities[i] for i in id_to_grid_coordinate_list ∩ targetidx]
    return Grid(
        g.size,
        g.costfunction,
        costmatrix,
        affinitymatrix,
        id_to_grid_coordinate_list,
        g.source_qualities,
        g.target_qualities,
        targetidx,
        targetnodes,
        qs,
        qt,
        g.dims,
    )
end

"""
    least_cost_distance(g::Grid)::Matrix{Float64}

Compute the least cost distance from all the cells in the grid to all target cells.

# Examples
```jldoctests
julia> affinities = [1/4 0 1/2 1/4
                     1/4 0 1/2 1/4
                     1/4 0 1/2 1/4
                     1/4 0 1/2 1/4];

julia> grid = ConScape.Grid(size(affinities)..., affinities=ConScape.graph_matrix_from_raster(affinities))
[ Info: cost graph contains 6 strongly connected subgraphs
[ Info: removing 8 nodes from affinity and cost graphs
ConScape.Grid of size 4x4

julia> ConScape.least_cost_distance(grid)
8×8 Matrix{Float64}:
 0.0       0.693147  1.38629   2.07944   0.693147  1.03972   1.73287   2.42602
 0.693147  0.0       0.693147  1.38629   1.03972   0.693147  1.03972   1.73287
 1.38629   0.693147  0.0       0.693147  1.73287   1.03972   0.693147  1.03972
 2.07944   1.38629   0.693147  0.0       2.42602   1.73287   1.03972   0.693147
 1.38629   1.73287   2.42602   3.11916   0.0       1.38629   2.77259   3.46574
 1.73287   1.38629   1.73287   2.42602   1.38629   0.0       1.38629   2.77259
 2.42602   1.73287   1.38629   1.73287   2.77259   1.38629   0.0       1.38629
 3.11916   2.42602   1.73287   1.38629   3.46574   2.77259   1.38629   0.0
```
"""
function least_cost_distance(g::Grid; θ::Nothing=nothing, approx::Bool=false)
    # FIXME! This should be multithreaded. However, ProgressLogging currently
    # does not support multithreading
    if approx
        throw(ArgumentError("no approximate algorithm is available for this distance function"))
    end
    targets = g.targetidx
    @progress vec_of_vecs = [least_cost_distance(g, target) for target in targets]

    return reduce(hcat, vec_of_vecs)
end
function least_cost_distance(g::Grid, target::CartesianIndex{2})
    graph = SimpleWeightedDiGraph(g.costmatrix)
    targetnode = findfirst(isequal(target), g.id_to_grid_coordinate_list)
    distvec = dijkstra_shortest_paths(graph, targetnode).dists
    return distvec
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
    coarse_graining(g.target_qualities, npix;
        id_to_grid_coordinate_list=g.id_to_grid_coordinate_list
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
    id_to_grid_coordinate_list=_id_gc_list(size(M)...)
)
    nrows, ncols = size(M)
    getrows = (floor(Int, npix / 2)+1):npix:(nrows-ceil(Int, npix / 2)+1)
    getcols = (floor(Int, npix / 2)+1):npix:(ncols-ceil(Int, npix / 2)+1)
    coarse_target_rc = Base.product(getrows, getcols)
    coarse_target_ids = vec(
        [
        findfirst(
            isequal(CartesianIndex(ij)),
            id_to_grid_coordinate_list
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
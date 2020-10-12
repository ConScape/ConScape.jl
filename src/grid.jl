abstract type Transformation end
struct MinusLog     <: Transformation end
struct ExpMinus     <: Transformation end
struct Inv          <: Transformation end
struct OddsAgainst  <: Transformation end
struct OddsFor      <: Transformation end

(::MinusLog)(x::Number)     = -log(x)
(::ExpMinus)(x::Number)     = exp(-x)
(::Inv)(x::Number)          = inv(x)
(::OddsAgainst)(x::Number)  = inv(x) - 1
(::OddsFor)(x::Number)      = x/(1 - x)

Base.inv(::MinusLog)     = ExpMinus()
Base.inv(::ExpMinus)     = MinusLog()
Base.inv(::Inv)          = Inv()
Base.inv(::OddsAgainst)  = OddsFor()
Base.inv(::OddsFor)      = OddsAgainst()

struct Grid
    nrows::Int
    ncols::Int
    affinities::SparseMatrixCSC{Float64,Int}
    costfunction::Union{Nothing,Transformation}
    costmatrix::SparseMatrixCSC{Float64,Int}
    id_to_grid_coordinate_list::Vector{CartesianIndex{2}}
    source_qualities::Matrix{Float64}
    target_qualities::AbstractMatrix{Float64}
end

"""
    Grid(nrows::Integer,
              ncols::Integer;
              affinities=nothing,
              qualities::Matrix=ones(nrows, ncols),
              source_qualities::Matrix=qualities,
              target_qualities::AbstractMatrix=qualities,
              costs::Union{Transformation,SparseMatrixCSC{Float64,Int}}=MinusLog(),
              prune=true)::Grid

Construct a `Grid` from an `affinities` matrix of type `SparseMatrixCSC`. It is possible
to also supply matrices of `source_qualities` and `target_qualities` as well as
a `costs` function that maps the `affinities` matrix to a `costs` matrix. Alternatively,
it is possible to supply a matrix to `costs` directly. If `prune=true` (the default), the
affinity and cost matrices will be pruned to exclude unreachable nodes.
"""
function Grid(nrows::Integer,
              ncols::Integer;
              affinities=nothing,
              qualities::Matrix=ones(nrows, ncols),
              source_qualities::Matrix=qualities,
              target_qualities::AbstractMatrix=qualities,
              costs::Union{Transformation,SparseMatrixCSC{Float64,Int}}=MinusLog(),
              prune=true)

    if affinities === nothing
        throw(ArgumentError("matrix of affinities must be supplied"))
    end

    if nrows*ncols != LinearAlgebra.checksquare(affinities)
        n = size(affinities, 1)
        throw(ArgumentError("grid size ($nrows, $ncols) is incompatible with size of affinity matrix ($n, $n)"))
    end

    _source_qualities = convert(Matrix{Float64}        , source_qualities)
    _target_qualities = convert(AbstractMatrix{Float64}, target_qualities)

    # Prune
    # id_to_grid_coordinate_list = if prune
    #     nonzerocells = findall(!iszero, vec(sum(affinities, dims=1)))
    #     _affinities = affinities[nonzerocells, nonzerocells]
    #     vec(CartesianIndices((nrows, ncols)))[nonzerocells]
    # else
    #     _affinities = affinities
    #     vec(CartesianIndices((nrows, ncols)))
    # end
    id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))

    costfunction, costmatrix = if costs isa Transformation
        costs, mapnz(costs, affinities)
    else
        if nrows*ncols != LinearAlgebra.checksquare(costs)
            n = size(costs, 1)
            throw(ArgumentError("grid size ($nrows, $ncols) is incompatible with size of cost matrix ($n, $n)"))
        end
        nothing, costs
    end

    if any(t -> t < 0, nonzeros(costmatrix))
        throw(ArgumentError("The cost graph can have only non-negative edge weights. Perhaps you should change the cost function?"))
    end

    if ne(difference(SimpleDiGraph(costmatrix), SimpleDiGraph(affinities))) > 0
        throw(ArgumentError("cost graph contains edges not present in the affinity graph"))
    end

    g = Grid(
        nrows,
        ncols,
        affinities,
        costfunction,
        costmatrix,
        id_to_grid_coordinate_list,
        _source_qualities,
        _target_qualities,
    )

    if prune
        return largest_subgraph(g)
    else
        return g
    end
end

Base.size(g::Grid) = (g.nrows, g.ncols)

function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    print(io, summary(g), " of size ", g.nrows, "x", g.ncols)
end

function Base.show(io::IO, ::MIME"text/html", g::Grid)
    t = string(summary(g), " of size ", g.nrows, "x", g.ncols)
    write(io, "<h4>$t</h4>")
    write(io, "<table><tr><td>Affinities</br>")
    show(io, MIME"text/html"(), plot_outdegrees(g))
    write(io, "</td></tr></table>")
    if g.source_qualities === g.target_qualities
        write(io, "<table><tr><td>Qualities</td></tr></table>")
        show(io, MIME"text/html"(), heatmap(g.source_qualities, yflip=true))
    else
        write(io, "<table><tr><td>Source qualities")
        show(io, MIME"text/html"(), heatmap(g.source_qualities, yflip=true))
        write(io, "</td><td>Target qualities")
        show(io, MIME"text/html"(), heatmap(Matrix(g.target_qualities), yflip=true))
        write(io, "</td></tr></table>")
    end
end

# Compute a vector of the cartesian indices of nonzero target qualities and
# the corresponding node id corresponding to the indices
_targetidx(q::Matrix, grididxs::Vector) = grididxs
_targetidx(q::SparseMatrixCSC, grididxs::Vector) =
    CartesianIndex.(findnz(q)[1:2]...) ∩ grididxs

function _targetidx_and_nodes(g::Grid)
    targetidx = _targetidx(g.target_qualities, g.id_to_grid_coordinate_list)
    targetnodes = findall(
        t -> t ∈ targetidx,
        g.id_to_grid_coordinate_list)
    return targetidx, targetnodes
end

function plot_values(g::Grid, values::Vector; kwargs...)
    canvas = fill(NaN, g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    heatmap(canvas, yflip=true, axis=nothing, border=:none, aspect_ratio=:equal; kwargs...)
end

function plot_outdegrees(g::Grid; kwargs...)
    values = sum(g.affinities, dims=2)
    canvas = zeros(g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    heatmap(canvas, yflip=true, axis=nothing, border=:none; kwargs...)
end

function plot_indegrees(g::Grid; kwargs...)
    values = sum(g.affinities, dims=1)
    canvas = zeros(g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    heatmap(canvas, yflip=true, axis=nothing, border=:none; kwargs...)
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
LightGraphs.is_strongly_connected(g::Grid) = is_strongly_connected(SimpleWeightedDiGraph(g.affinities))

"""
    largest_subgraph(g::Grid)::Grid

Extract the largest fully connected subgraph of the `Grid`. The returned `Grid`
will have the same size as the input `Grid` but only nodes associated with the
largest subgraph of the affinities will be active.
"""
function largest_subgraph(g::Grid)
    # Convert cost matrix to graph
    graph = SimpleWeightedDiGraph(g.costmatrix, permute=false)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    @info "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])

    ndiffnodes = size(g.costmatrix, 1) - length(scci)
    if ndiffnodes > 0
        @info "removing $ndiffnodes nodes from affinity and cost graphs"
    end

    # Extract the adjacency matrix of the largest subgraph
    affinities = g.affinities[scci, scci]
    # affinities = convert(SparseMatrixCSC{Float64,Int}, graph[scci])

    return Grid(
        g.nrows,
        g.ncols,
        affinities,
        g.costfunction,
        g.costfunction === nothing ? g.costmatrix[scci, scci] : mapnz(g.costfunction, affinities),
        g.id_to_grid_coordinate_list[scci],
        g.source_qualities,
        g.target_qualities)
end

"""
    least_cost_distance(g::Grid, target::Tuple{Int,Int})::Matrix{Float64}

Compute the least cost distance from all the cells in the grid to the the `target` cell.

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

julia> ConScape.least_cost_distance(grid, (4,4))
4×4 Array{Float64,2}:
 Inf  Inf  2.42602   3.46574
 Inf  Inf  1.73287   2.77259
 Inf  Inf  1.03972   1.38629
 Inf  Inf  0.693147  0.0
```
"""
function least_cost_distance(g::Grid, target::Tuple{Int,Int})
    graph = SimpleWeightedDiGraph(g.costmatrix)
    targetnode = findfirst(isequal(CartesianIndex(target)), g.id_to_grid_coordinate_list)
    distvec = dijkstra_shortest_paths(graph, targetnode).dists
    distgrid = fill(Inf, g.nrows, g.ncols)
    for (i, c) in enumerate(g.id_to_grid_coordinate_list)
        distgrid[c] = distvec[i]
    end
    return distgrid
end


"""
    sum_neighborhood(g::Grid, rc::Tuple{Int,Int}, npix::Integer)::Float64

A helper-function, used by coarse_graining, that computes the sum of pixels within a npix neighborhood around the target rc.
"""
function sum_neighborhood(g, rc, npix)
    getrows = (rc[1] - floor(Int, npix/2)):(rc[1] + (ceil(Int, npix/2) - 1))
    getcols = (rc[2] - floor(Int, npix/2)):(rc[2] + (ceil(Int, npix/2) - 1))
    return sum(g.target_qualities[getrows, getcols])
end



"""
    coarse_graining(g::Grid, npix::Integer)::Array

Creates a sparse matrix of target qualities for the landmarks based on merging npix pixels into the center pixel.
"""
function coarse_graining(g, npix)
    getrows = (floor(Int, npix/2)+1):npix:(g.nrows-ceil(Int, npix/2)+1)
    getcols = (floor(Int, npix/2)+1):npix:(g.ncols-ceil(Int, npix/2)+1)
    coarse_target_rc = Base.product(getrows, getcols)
    coarse_target_ids = vec([findfirst(isequal(CartesianIndex(ij)),
                                       g.id_to_grid_coordinate_list) for ij in coarse_target_rc])
    coarse_target_rc = [ij for ij in coarse_target_rc if !ismissing(ij)]
    filter!(!ismissing, coarse_target_ids)
    V = [sum_neighborhood(g, ij, npix) for ij in coarse_target_rc]
    I = first.(coarse_target_rc)
    J = last.(coarse_target_rc)
    target_mat = sparse(I, J, V, g.nrows, g.ncols)
    target_mat = dropzeros(target_mat)

    return target_mat
end

mutable struct Grid
    nrows::Int
    ncols::Int
    A::SparseMatrixCSC{Float64,Int}
    id_to_grid_coordinate_list::Vector{CartesianIndex{2}}
    source_qualities::Matrix{Float64}
    target_qualities::Matrix{Float64}
end

"""
    Grid(nrows::Integer, ncols::Integer;
              qualities::Matrix=ones(nrows, ncols),
              source_qualities::Matrix=qualities,
              target_qualities::Matrix=qualities,
              nhood_size::Integer=8,
              landscape=_generateA(nrows, ncols, nhood_size))::Grid

Construct a `Grid` from a `landscape` passed a `SparseMatrixCSC`.
"""
function Grid(nrows::Integer,
              ncols::Integer;
              qualities::Matrix=ones(nrows, ncols),
              source_qualities::Matrix=qualities,
              target_qualities::Matrix=qualities,
              nhood_size::Integer=8,
              landscape=_generateA(nrows, ncols, nhood_size),
              prune=true)

    @assert nrows*ncols == LinearAlgebra.checksquare(landscape)
    Ngrid = nrows*ncols

    if source_qualities === target_qualities
        _source_qualities = _target_qualities = convert(Matrix{Float64}, source_qualities)
    else
        _source_qualities = convert(Matrix{Float64}, source_qualities)
        _target_qualities = convert(Matrix{Float64}, target_qualities)
    end

    # Prune
    if prune
        nonzerocells = findall(!iszero, vec(sum(landscape, dims=1)))
        _landscape = landscape[nonzerocells, nonzerocells]
        _id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))[nonzerocells]
    else
        _landscape = landscape
        _id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))
    end

    Grid(
        nrows,
        ncols,
        _landscape,
        _id_to_grid_coordinate_list,
        _source_qualities,
        _target_qualities
    )
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
        show(io, MIME"text/html"(), heatmap(g.target_qualities, yflip=true))
        write(io, "</td></tr></table>")
    end
end

function plot_outdegrees(g::Grid)
    values = sum(g.A, dims=2)
    canvas = zeros(g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    heatmap(canvas, yflip=true)
end

"""
    is_connected(g::Grid)::Bool

Test if graph defined by Grid is fully connected.

# Examples

```jldoctests
julia> landscape = [1/4 0 1/4 1/4
                    1/4 0 1/4 1/4
                    1/4 0 1/4 1/4
                    1/4 0 1/4 1/4];

julia> grid = ConScape.Grid(size(landscape)..., landscape=ConScape.adjacency(landscape))
ConScape.Grid of size 4x4

julia> ConScape.is_connected(grid)
false
```
"""
LightGraphs.is_connected(g::Grid) = is_connected(SimpleWeightedDiGraph(g.A))

"""
    largest_subgraph(g::Grid)::Grid

Extract the largest fully connected subgraph of the `Grid`. The returned `Grid`
will have the same size as the input `Grid` but only nodes associated with the
largest subgraph of the landscape will be active.
"""
function largest_subgraph(g::Grid)
    # Convert adjacency matrix to graph
    graph = SimpleWeightedDiGraph(g.A, permute=false)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])

    # Extract the adjacency matrix of the largest subgraph
    newA = graph[scci]

    return Grid(g.nrows, g.ncols, newA, g.id_to_grid_coordinate_list[scci], g.source_qualities, g.target_qualities)
end

"""
    least_cost_distance(g::Grid, target::Tuple{Int,Int})::Matrix{Float64}

Compute the least cost distance from all the cells in the grid to the the `target` cell.

# Examples
```jldoctests
julia> landscape = [1/4 0 1/4 1/4
                    1/4 0 1/4 1/4
                    1/4 0 1/4 1/4
                    1/4 0 1/4 1/4];

julia> grid = ConScape.Grid(size(landscape)..., landscape=ConScape.adjacency(landscape))
ConScape.Grid of size 4x4

julia> ConScape.least_cost_distance(grid, (4,4))
4×4 Array{Float64,2}:
 Inf  NaN  0.75  0.75
 Inf  NaN  0.5   0.5
 Inf  NaN  0.25  0.25
 Inf  NaN  0.25  0.0
```
"""
function least_cost_distance(g::Grid, target::Tuple{Int,Int})
    graph = SimpleWeightedDiGraph(g.A)
    node = findfirst(isequal(CartesianIndex(target)), g.id_to_grid_coordinate_list)
    distvec = dijkstra_shortest_paths(graph, node).dists
    distgrid = fill(NaN, g.nrows, g.ncols)
    for (i, c) in enumerate(g.id_to_grid_coordinate_list)
        distgrid[c] = distvec[i]
    end
    return distgrid
end

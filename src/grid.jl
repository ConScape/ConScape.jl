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
              landscape=_generateA(nrows, ncols, nhood_size)) -> Grid

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
        show(io, MIME"text/html"(), heatmap(g.source_qualities))
    else
        write(io, "<table><tr><td>Source qualities")
        show(io, MIME"text/html"(), heatmap(g.source_qualities))
        write(io, "</td><td>Target qualities")
        show(io, MIME"text/html"(), heatmap(g.target_qualities))
        write(io, "</td></tr></table>")
    end
end

function plot_outdegrees(g::Grid)
    values = sum(g.A, dims=2)
    canvas = zeros(g.nrows, g.ncols)
    for (i,v) in enumerate(values)
        canvas[g.id_to_grid_coordinate_list[i]] = v
    end
    heatmap(canvas)
end

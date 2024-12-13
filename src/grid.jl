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
    source_qualities::AbstractMatrix{Float64}
    target_qualities::AbstractMatrix{Float64}
    dims::Union{Tuple,Nothing}
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
              qualities::AbstractMatrix=ones(nrows, ncols),
              source_qualities::AbstractMatrix=qualities,
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

    _source_qualities = convert(Matrix{Float64}        , _unwrap(source_qualities))
    _target_qualities = convert(AbstractMatrix{Float64}, _unwrap(target_qualities))

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
        dims(source_qualities),
    )

    if prune
        return largest_subgraph(g)
    else
        return g
    end
end

Base.size(g::Grid) = (g.nrows, g.ncols)
DimensionalData.dims(g::Grid) = g.dims

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
_unwrap(R::Raster) = parent(R)
_unwrap(R::AbstractMatrix) = R
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

function _fill_matrix(values, g) 
    M = fill(NaN, g.nrows, g.ncols)
    for (i, v) in enumerate(values)
        M[g.id_to_grid_coordinate_list[i]] = v
    end
    return M
end

function Raster(values::AbstractVector, g::Grid; kwargs...)
    isnothing(dims(g)) && throw(ArgumentError("Grid dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, g), dims(g); kwargs...)
end

function outdegrees(g::Grid)
    values = sum(g.affinities, dims=2)
    _maybe_raster(_fill_matrix(values, g), g)
end

function indegrees(g::Grid; kwargs...)
    values = sum(g.affinities, dims=1)
    _maybe_raster(_fill_matrix(values, g), g)
end

plot_values(g::Grid, values::Vector; kwargs...) =
    _heatmap(_fill_matrix(values, g), g; kwargs...)
plot_outdegrees(g::Grid; kwargs...) = _heatmap(outdegrees(g), g; kwargs...)
plot_indegrees(g::Grid; kwargs...) = _heatmap(indegrees(g), g; kwargs...)


# If the grid has raster dimensions, 
# plot as a raster on a spatial grid
function _heatmap(canvas, g; kwargs...)
    if isnothing(dims(g))
        heatmap(canvas; yflip=true, axis=nothing, border=:none, aspect_ratio=:equal, kwargs...)
    else
        heatmap(Raster(canvas, dims(g)); kwargs...)
    end
end

function assess(g::Grid) 
    targetidx, targetnodes = _targetidx_and_nodes(g)
    # Calculate memory use and expected flops for 
    # targetnodes, or something...
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
Graphs.is_strongly_connected(g::Grid) = is_strongly_connected(SimpleWeightedDiGraph(g.affinities))

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
        g.target_qualities,
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
    targets = ConScape._targetidx_and_nodes(g)[1]
    @progress vec_of_vecs = [_least_cost_distance(g, target) for target in targets]

    return reduce(hcat, vec_of_vecs)
end

function _least_cost_distance(g::Grid, target::CartesianIndex{2})
    graph = SimpleWeightedDiGraph(g.costmatrix)
    targetnode = findfirst(isequal(target), g.id_to_grid_coordinate_list)
    distvec = dijkstra_shortest_paths(graph, targetnode).dists
    return distvec
end

function _vec_to_grid(g::Grid, vec::Vector)
    grid = fill(Inf, g.nrows, g.ncols)
    for (i, c) in enumerate(g.id_to_grid_coordinate_list)
        grid[c] = vec[i]
    end
    return grid
end

"""
    sum_neighborhood(g::Grid, rc::Tuple{Int,Int}, npix::Integer)::Float64

A helper-function, used by coarse_graining, that computes the sum of pixels within a npix neighborhood around the target rc.
"""
function sum_neighborhood(g, rc, npix)
    getrows = (rc[1] - floor(Int, npix/2)):(rc[1] + (ceil(Int, npix/2) - 1))
    getcols = (rc[2] - floor(Int, npix/2)):(rc[2] + (ceil(Int, npix/2) - 1))
    # pixels outside of the landscape are encoded with NaNs but we don't want
    # the NaNs to propagate to the coarse grained values
    return sum(t -> isnan(t) ? 0.0 : t, g.target_qualities[getrows, getcols])
end



"""
    coarse_graining(g::Grid, npix::Integer)::Array

Creates a sparse matrix of target qualities for the landmarks based on merging npix pixels into the center pixel.
"""
function coarse_graining(g, npix)
    getrows = (floor(Int, npix/2)+1):npix:(g.nrows-ceil(Int, npix/2)+1)
    getcols = (floor(Int, npix/2)+1):npix:(g.ncols-ceil(Int, npix/2)+1)
    coarse_target_rc = Base.product(getrows, getcols)
    coarse_target_ids = vec(
        [
            findfirst(
                isequal(CartesianIndex(ij)),
                g.id_to_grid_coordinate_list
            ) for ij in coarse_target_rc
        ]
    )
    coarse_target_rc = [ij for ij in coarse_target_rc if !ismissing(ij)]
    filter!(!ismissing, coarse_target_ids)
    V = [sum_neighborhood(g, ij, npix) for ij in coarse_target_rc]
    I = first.(coarse_target_rc)
    J = last.(coarse_target_rc)
    target_mat = sparse(I, J, V, g.nrows, g.ncols)
    target_mat = dropzeros(target_mat)

    return target_mat
end


"""
    free_energy_distance(
        g::Grid;
        θ::Union{Real,Nothing}=nothing,
        approx::Bool=false
    )

Compute the randomized shorted path based expected costs from all source nodes to
all target nodes in the graph defined by `g` using the inverse temperature parameter
`θ`. The computation can either continue until convergence when setting `approx=false`
(the default) or return an approximate result based on just a single iteration of the Bellman-Ford
algorithm when `approx=true`.
"""
function expected_cost(
    g::Grid;
    θ::Union{Real,Nothing}=nothing,
    approx::Bool=false
)
    # FIXME! This should be multithreaded. However, ProgressLogging currently
    # does not support multithreading
    targets = ConScape._targetidx_and_nodes(g)[1]
    @progress vec_of_vecs = [_expected_cost(g, target, θ, approx) for target in targets]

    return reduce(hcat, vec_of_vecs)
end

function _expected_cost(
    g::Grid,
    target::CartesianIndex{2},
    θ::Union{Nothing,Real},
    approx::Bool
)
    if θ === nothing || θ <= 0
        throw(ArgumentError("θ must be a positive number"))
    end

    Pref = _Pref(g.affinities)

    targetid = searchsortedfirst(g.id_to_grid_coordinate_list, target)

    return first(bellman_ford(Pref, g.costmatrix, θ, targetid, approx))
end


"""
    free_energy_distance(
        g::Grid;
        target::Union{Tuple{Int,Int},Nothing}=nothing,
        θ::Union{Real,Nothing}=nothing,
        approx::Bool=false
    )

Compute the directed free energy distance from all source nodes to
all target nodes in the graph defined by `g` using the inverse temperature parameter
`θ`. The computation can either continue until convergence when setting `approx=false`
(the default) or return an approximate result based on just a single iteration of the Bellman-Ford
algorithm when `approx=true`.
"""
function free_energy_distance(
    g::Grid;
    θ::Union{Real,Nothing}=nothing,
    approx::Bool=false
)
    # FIXME! This should be multithreaded. However, ProgressLogging currently
    # does not support multithreading
    targets = ConScape._targetidx_and_nodes(g)[1]
    @progress vec_of_vecs = [_free_energy_distance(g, target, θ, approx) for target in targets]

    return reduce(hcat, vec_of_vecs)
end

function _free_energy_distance(
    g::Grid,
    target::CartesianIndex{2},
    θ::Union{Real,Nothing},
    approx::Bool
)
    if θ === nothing || θ <= 0
        throw(ArgumentError("θ must be a positive number"))
    end

    Pref = _Pref(g.affinities)

    targetid = searchsortedfirst(g.id_to_grid_coordinate_list, target)

    return last(bellman_ford(Pref, g.costmatrix, θ, targetid, approx))
end

survival_probability(
    g::Grid;
    θ::Union{Real,Nothing}=nothing,
    approx::Bool=false
) = exp.((-).(free_energy_distance(g; θ=θ, approx=approx) .* θ))

power_mean_proximity(
    g::Grid;
    θ::Union{Real,Nothing}=nothing,
    approx::Bool=false
) = survival_probability(g; θ=θ, approx=approx) .^ (1/θ)

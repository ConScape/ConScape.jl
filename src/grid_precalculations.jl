"""
    GridPrecalculations

Abstract type for precalculated variables for use in graph measures.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables once for all targets of multiple
graph measure.
"""
abstract type GridPrecalculations end

grid(gp::GridPrecalculations) = gp.grid
Base.size(gp::GridPrecalculations) = size(grid(gp))
DimensionalData.dims(gp::GridPrecalculations) = dims(grid(gp))

"""
    precalculate(::MovementMode, g::Grid)

Returns a `GridPrecalculation` object for the specific `MovementMode`.

This contains all sparse and dense matrices needed for computing 
graph measures for the targets in the `Grid` `g`.
"""
function precalculate end

"""
    RandomisedShortestPathPrecalculations(g::Grid; θ=nothing)

Stores precalculated variables for use in `RandomisedShortestPath`-based measures.

(formerly GridRSP)
"""
struct RandomisedShortestPathPrecalculations <: GridPrecalucations
    g::Grid
    θ::Float64
    probability::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    fundamental::Matrix{Float64}
    # TODO the rest here
end
function RandomisedShortestPathPrecalculations(g::Grid; θ=nothing, verbose=true)
    Pref = _Pref(g.affinities)
    W = _W(Pref, θ, g.costmatrix)

    Z = (I - W) \ Matrix(sparse(g.targetnodes,
        1:length(g.targetnodes),
        1.0,
        size(g.costmatrix, 1),
        length(g.targetnodes)))
    # Check that values in Z are not too small:
    verbose && if minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end

    return RandomisedShortestPathPrecalculations(g, θ, Pref, W, Z)
end

precalculate(m::RandomisedShortestPath, g) = RandomisedShortestPathPrecalculations(g; θ=m.θ)

function reinit!(allocs::RandomisedShortestPathPrecalculations, g; θ=nothing)
end

"""
    LeastCostPrecalculations(g::Grid)

Stores precalculated variables for use in `LeastCost`-based measures.
"""
struct LeastCostPrecalculations <: GridPrecalucations
    g::Grid
    probability::SparseMatrixCSC{Float64,Int}
    cost_weighted_digraph::SimpleWeightedDiGraph{Float64}
    # TODO the rest here
end
function LeastCostPrecalculations(g)
    probability = _Pref(g.affinities)
    cost_weighted_digraph = simpleweighteddigraph(g.costmatrix)
    LeastCostPrecalculations(g, probability, cost_weighted_digraph)
end

precalculate(::LeastCost, g) = LeastCostPrecalculations(g)

function reinit!(allocs::LeastCostPrecalculations, g)
end

"""
    RandomWalkPrecalculations(g::Grid)

Stores precalculated variables for use in `RandomWalk`-based measures.
"""
struct RandomWalkPrecalculations <: GridPrecalucations
    g::Grid
    probability::SparseMatrixCSC{Float64,Int}
    fundamental::Matrix{Float64}
    hitting_time::Matrix{Float64}
    stationary_distribution::Vector{Float64}
    # TODO the rest here
end
function RandomWalkPrecalculations(g)
    probability = _Pref(g.affinities)
    p = stationary_distribution(probability)
    fundamental = inv(Matrix(I-P) .+ p')
    hitting_time = (diag(Z)' .- Z) ./ p'
    RandomWalkPrecalculations(g, probability, fundamental, hitting_time, p)
end

precalculate(::RandomWalk, g) = RandomWalkPrecalculations(g)

function reinit!(allocs::RandomWalkPrecalculations, g)
    # TODO in-place version
end
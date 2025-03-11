#=
`compute` and `compute_target` for GraphMeasures.
eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

# Compute can be called on any graph measure
compute(m::MovementMode, gm::GraphMeasure, g::Grid) = 
    compute(returntrait(gm) m, gm, g)
# We specialise on returntrait
function compute(::ReturnsDenseSpatial, m::MovementMode, gm::GraphMeasure, g::Grid)
    grid_precalculations = precalculate(m, g)
    output = allocate_output(gm, g)
    for (i, t) in enumerate(g.targetids)
        v = compute_target(m, gm, grid_precalculations, t)
        output[g.id_to_grid_coordinate_list[i]] = v
    end
    return output
end
function compute(::ReturnsSparse, m::MovementMode, gm::GraphMeasure, g::GridPrecalculations)
    output = allocate_output(gm, g)
    for (i, t) in enumerate(g.targetids)
        v = compute_target(m, gm, g, t)
        # TODO output[...] = v
    end
    return output
end

"""
    compute_target(::MovementMode, gm::GraphMeasure, g::GridPrecalculations, target::Int)

Computes results of a single target pixel for a graph measure
and movement mode. 

`compute_target` is called inside `compute` or in the loop in `solve!` for 
VectorSolver/LinearSolver. 
"""
function compute_target end

# LeastCost
function compute_target(::LeastCost, gm::Betweenness, g::GridPrecalculations, target::Int)
    # Calculate distances
    (; cost_weighted_digraph) = g.cost_weighted_digraph # simpleweighteddigraph(g.costmatrix)
    shorted_paths = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, target)
    shortest_paths_en = Graphs.enumerate_paths(shorted_paths)
    k = ones(size(g.costmatrix)[1]) # TODO use a workspace
    # And maybe transform them
    if !isnothing(distance_transformation(cm))
        k .= distance_transformation(cm).(shorted_paths.dists)
    end

    # Apply weight for the specific BetweennessWeight
    apply_weight!(k, gm, g, target)

    # TODO what does all this do...
    shortest_paths_en[target] = [target]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)

    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw * k
end

# RandomWalk 
function compute_target(m::RandomWalk, ::EdgeBetweenness{Weighting}, g::GridPrecalculations, (i, t)) where Weighting
    nodebet = compute_target(m, Betweenness(Weighting()), g, t)
    return nodebet * pref[g.id_to_grid_coordinate_list[i]]
end
function compute_target(::RandomWalk, ::Betweenness{QualityWeighted}, g::GridPrecalculations, t::Int)
    Z, H, qˢ, qᵗ, p, workspace = g.fundamental, g.hitting_time, g.source_quality, g.target_quality, g.stationary_distribution, g.workspaces[1]
    return qˢ' * _betweenness!(workspace, Z, H, p, t) * qᵗ
end
function compute_target(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}, g::GridPrecalculations, target::Int)
    Z, H, K, p, workspace = g.probability, g.cost, g.fundamental, g.hitting_time, g.quality_weighted_proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end
function compute_target(::RandomWalk, ::Betweenness{ProximityWeighted}, g::GridPrecalculations, target::Int)
    Z, H, K, p, workspace = g.fundamental, g.hitting_time, g.proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end

# TODO this only works for simmetrical Z
_betweenness!(workspace, Z, H, p, t) = 
    workspace .= view(Z, :, t) .- view(Z, :, t)' .+ H .* p[t]

# This is an idea for specifying precomputed arrays needed for a given graph measure
# It would be nice to have this close top the algorithme.
# This function could also be defined with a @needs macro on the 
# `compute_target` function to remove the name duplication
needs(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :source_quality, :target_quality, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :quality_weighted_proximity, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{ProximityWeighted}) = 
    (:fundamental, :hitting_time, :proximity, :stationary_distribution, :workspaces => 1)


function stationary_distribution(P::SparseMatrixCSC)
    #Input: the transition probability matrix P
    #Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI[1, :] = ones(n)
    v = [1; zeros(n - 1)]
    return PI \ v
end

apply_weight!(k, ::Unweigthed, g::GridPrecalculations, target) = k
function apply_weight!(k, ::QualityAndProximityWeighted, g::GridPrecalculations, target)
    k .*= g.qˢ .* g.qᵗ[findfirst(isequal(target), g.targetidx)]
    return k
end
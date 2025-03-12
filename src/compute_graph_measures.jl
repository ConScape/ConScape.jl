#=
`compute` and `compute_target` for GraphMeasures.
eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

# Compute can be called on any graph measure
compute(m::MovementMode, gm::GraphMeasure, g::Grid) = compute(returntrait(gm) m, gm, g)
# We specialise on returntrait
function compute(::ReturnsDenseSpatial, m::MovementMode, gm::GraphMeasure, g::Grid)
    grid_precalculations = precalculate(m, g)
    output = allocate_output(gm, g)
    for (i, t) in enumerate(g.targetids)
        v = compute(m, gm, grid_precalculations, t)
        output[g.id_to_grid_coordinate_list[i]] = v
    end
    return output
end
function compute(::ReturnsSparse, m::MovementMode, gm::GraphMeasure, g::GridPrecalculations)
    output = allocate_output(gm, g)
    for (i, t) in enumerate(g.targetids)
        v = compute(m, gm, g, t)
        # TODO output[...] = v
    end
    return output
end

"""
    compute(::MovementMode, gm::GraphMeasure, g::GridPrecalculations, target::Int)

Computes results of a single target pixel for a graph measure
and movement mode. 

`compute` is called inside `compute` or in the loop in `solve!` for 
VectorSolver/LinearSolver. 
"""
function compute end

# LeastCost
function compute(::LeastCost, gm::Betweenness, g::GridPrecalculations, ids::NamedTuple)
    # Calculate distances
    (; cost_weighted_digraph) = g.cost_weighted_digraph # simpleweighteddigraph(g.costmatrix)
    shorted_paths = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, ids.targetid)
    shortest_paths_en = Graphs.enumerate_paths(shorted_paths)
    k = ones(size(g.costmatrix)[1]) # TODO use a workspace
    # And maybe transform them
    if !isnothing(distance_transformation(cm))
        k .= distance_transformation(cm).(shorted_paths.dists)
    end

    # Apply weight for the specific BetweennessWeight
    apply_weight!(k, gm, g, ids.targetid)

    # TODO what does all this do...
    shortest_paths_en[ids.targetid] = [ids.targetid]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)

    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw * k
end

# RandomWalk 
function compute(m::RandomWalk, ::EdgeBetweenness{Weighting}, g::GridPrecalculations, ids::NamedTuple) where Weighting
    nodebet = compute(m, Betweenness(Weighting()), g, t)
    return nodebet * pref[ids.spatialid]
end
function compute(::RandomWalk, ::Betweenness{QualityWeighted}, g::GridPrecalculations, ids::NamedTuple)
    Z, H, qˢ, qᵗ, p, workspace = g.fundamental, g.hitting_time, g.source_quality, g.target_quality, g.stationary_distribution, g.workspaces[1]
    return qˢ' * _betweenness!(workspace, Z, H, p, ids) * qᵗ
end
function compute(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}, g::GridPrecalculations, ids::NamedTuple)
    Z, H, K, p, workspace = g.probability, g.cost, g.fundamental, g.hitting_time, g.quality_weighted_proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, ids) .* K)
end
function compute(::RandomWalk, ::Betweenness{ProximityWeighted}, g::GridPrecalculations, ids::NamedTuple)
    Z, H, K, p, workspace = g.fundamental, g.hitting_time, g.proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, ids) .* K)
end

# TODO this only works for simmetrical Z
function _betweenness!(workspace, Z, H, p, ids) 
    t = ids.targetid
    workspace .= view(Z, :, t) .- view(Z, :, t)' .+ H .* p[t]
end

# This is an idea for specifying precomputed arrays needed for a given graph measure
# It would be nice to have this close top the algorithme.
# This function could also be defined with a @needs macro on the 
# `compute` function to remove the name duplication
needs(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :source_quality, :target_quality, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :quality_weighted_proximity, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{ProximityWeighted}) = 
    (:fundamental, :hitting_time, :proximity, :stationary_distribution, :workspaces => 1)

apply_weight!(k, ::Unweigthed, g::GridPrecalculations, target) = k
function apply_weight!(k, ::QualityAndProximityWeighted, g::GridPrecalculations, target)
    # TODO lookup target not findfirst
    k .*= g.qˢ .* g.qᵗ[findfirst(isequal(target), g.targetidx)]
    return k
end

# RandomShortestPath

# Betweenness: TODO move code here from RSP and rewrite per-target
function compute(::RandomShortestPath, ::Betweenness{QualityWeighted}, gp::GridPrecalculations; kw...)
    g = grid(gp)
    return RSP_betweenness_qweighted(gp.W, gp.Z, g.qs, g.qt, g.targetnodes; kw...)
end
function compute(::RandomShortestPath, ::Betweenness{QualityAndProximityWeighted}, gp::GridPrecalculations; kw...)
    g = grid(gp)
    return RSP_betweenness_kweighted(gp.W, gp.Z, g.qs, g.qt, g.targetnodes; kw...)
end
function compute(::RandomShortestPath, ::EdgeBetweenness{QualityWeighted}, gp::GridPrecalculations; kw...) 
    g = gp.g
    return RSP_edge_betweenness_qweighted(gp.W, gp.Z, g.qs, g.qt, g.targetnodes; kw...)
end
function compute(::RandomShortestPath, ::EdgeBetweenness{QualityAndProximityWeighted}, gp::GridPrecalculations; kw...) 
    g = gp.g
    return RSP_edge_betweenness_kweighted(gp.W, gp.Z, g.qs, g.qt, g.targetnodes; kw...)
end
# ConnectedHabitat 
# the same for all movement modes once proximity is calcuated?
function compute(::MovementMode, ::ConnectedHabitat, gp::GridPrecalculations, t::Int)
    g = gp.g
    qˢ, qᵗ, K = g.source_quality, g.target_quality, gp.proximity
    return mul!(view(workspaces, :, 1), K, qᵗ) .*= qˢ
end

function compute(m::RandomisedShortestPath, gm::Sensitivity, gp::GridPrecalculations, (spatialid, targetid, targetnode)::Tuple{CartesianIndex{2},Int,Int})
    if gm.landscape_measure <: LandscapeEigen
        v, λ, w = compute(m, EigMax(), gp, (spatialid, targetid, targetnode))
        vTw = (v') * w
        if !(gm.context <: Quality)
            qˢ = qˢ .* v    
            qᵗ = qᵗ .* w
        end
    end

    target_sensitivity = if gm.context <: Union{Affinity,Cost,CostAndAffinitySensitivityContext} 
        S_e_aff, S_e_cost = if cm <: ExpectedCost
            diff_K_D = _diff_KD(distance_transformation)
            EC_sensitivity(g.affinities, g.costmatrix, m.θ, gp.W, gp.Z, diff_K_D, qˢ, qᵗ, [t])[1]
        else
            PM_sensitivity(g.affinities, nothing, m.θ, gp.W, gp.Z, gp.Z, qˢ, qᵗ, [t])[1]
        end

        if unitless 
            _scale!(S_e_aff, gm.context, g)
            _scale!(S_e_cost, gm.context, g)
        end

        if gm.context <: Affinity
            sum(S_e_aff)
        elseif gm.context <: Cost
            sum(S_e_cost)
        elseif gm.context <: AffinityAndCost
            diff_C_A = ConScape.mapnz(diff_C_A_fun, g.affinities)
            S_e_total = S_e_aff .+ S_e_cost .* diff_C_A
            sum(S_e_total)
        elseif gm.context <: CostAndAffinity
            diff_A_C = ConScape.mapnz(diff_A_C_fun, g.affinities)
            S_e_total = S_e_aff .* diff_A_C .+ S_e_cost
            sum(S_e_total)
        end
    elseif gm.context <: Qualities
        K = gp.proximities
        if landscape_measure <: LandscapeEigen
            K = v .* K .* (w')
        end
        K = K .+= transpose(K)
        target_sensitivity = K * g.target_qualities[I]
        if unitless
            target_sensitivity *= g.source_qualities[I]
        end
        target_sensitivity
    end

    if gm.landscape_measure <: LandscapeEigen
        target_sensitivity /= vTw
    end

    output[I] = target_sensitivity * isnan(g.source_qualities[I]) ? NaN : 1
    return output
end

_diff_CA_fun(::MinusLog) = x -> -inv(x)
_diff_AC_fun(::MinusLog) = x -> -(x)
_diff_CA_fun(::Inv) = x -> -inv(x^2) # TODO fix
_diff_AC_fun(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = -K .* x.α
_diff_KD(::Inv) = -K .^ 2

# TODO a more specific name
_scale!(A, ::Union{Affinity,AffinityAndCost}, g) = A .*= g.affinities
_scale!(A, ::Union{Cost,CostAndAffinity}, g) = A .*= g.costmatrix

function compute(::RandomisedShortestPath, gm::EigMax, gp::RandomisedShortestPathPrecalculations, (spatialid, targetid, targetnode))
    g = gp.g
    qSq = g.quality_scaled_proximity
    workspace1 = gp.workspaces
    vˡ = zeros(n)
    vʳ = fill(NaN, n)

    vʳ .= NaN
    # Square submatrix defined by extracting the rows corresponding to landmarks
    qSq₀₀ = view(workspace1, 1, :)
    qSq₀₀ .= view(qSq, t:t, :) # TODO: does this need to be a matrix?

    # Size of the full problem
    n = size(g.affinities, 1)

    # Use an Arnoldi based eigensolver to compute the largest (absolute) eigenvalue and right vector (of submatrix)
    Fps = partialschur(qSq₀₀; nev=1, tol=gm.tol)
    λ₀, vʳ₀ = partialeigen(Fps[1])

    # Construct full right vector
    vʳ[t] = vʳ₀
    for i in 1:n
        i == targetnode && continue
        vʳ[i] = view(qSq, i, :) * vʳ₀ / λ₀[1]
    end

    # Compute left vector (of submatrix) by shift-invert
    # TODO rewrite for single target
    Flu = lu(qSq₀₀ - λ₀[1] * I)
    vˡ₀ = ldiv!(Flu', rand(1))
    rmul!(vˡ₀, inv(vˡ₀[1]))
    vˡ[targetnode] = vˡ₀

    return (vˡ, λ₀=λ₀[1], vʳ)
end        
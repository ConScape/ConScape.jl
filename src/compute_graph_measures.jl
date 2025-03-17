#=
`compute` and `compute_target` for GraphMeasures.  eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

# ConnectedHabitat 
compute(::ConnectedHabitat, tp::TargetPrecalculations) = tp.M

"""
    compute(::MovementMode, gm::GraphMeasure, g::GridPrecalculations[, target::Int)

Computes results of a single target pixel for a graph measure
and movement mode. 

`compute` is called inside `compute` or in the loop in `solve!` for 
VectorSolver/LinearSolver. 
"""
function compute end


# Betweenness

# LeastCost
function compute(gm::Betweenness, tp::LeastCostTargetPrecalculations)
    # Calculate distances
    (; cost_weighted_digraph) = tp
    shorted_paths = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, target(tp).spatial)
    shortest_paths_en = Graphs.enumerate_paths(shorted_paths)

    # And maybe transform them
    # TODO untangle this logic so apply_weight handles everythign
    K = if isnothing(distance_transformation(tp))
        workspace1 .= 1.0 # TODO is this right? not dists?
    else
        workspace1 .= distance_transformation(cm).(shorted_paths.dists)
    end

    _apply_weight!(K, gm, tp)

    # TODO what does all this do...
    shortest_paths_en[targetid] = [targetid]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)
    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw * K
end

# _apply_weight!(K, ::Unweighted, tp::LeastCostTargetPrecalculations) = k
_apply_weight!(K, ::ProximityWeighted, tp::LeastCostTargetPrecalculations) = K
function _apply_weight!(K, ::QualityAndProximityWeighted, tp::LeastCostTargetPrecalculations)
    (; qˢ, qᵗ) = tp
    K .*= qˢ .* qᵗ
end

# RandomWalk 
function compute(::EdgeBetweenness{Weighting}, tp::RandomWalkTargetPrecalculations) where Weighting
    nodebet = compute(Betweenness(Weighting()), tp)
    return nodebet * pref[target(tp).spatial] # TODO get pref
end
function compute(bet::Betweenness, tp::RandomWalkTargetPrecalculations)
    (; Z, H, p, workspace) = tp
    node = target(tp).node
    # TODO not square
    workspace .= view(Z, :, 1) .- view(Z, :, node)' .+ H .* p[node]
    _apply_weight!(workspace, wieghting(bet), tp)
end

# TODO: these should be the same for all movement modes
_apply_weight!(ZZHp, ::Unweighted, tp::RandomWalkTargetPrecalculations) = sum(ZZHp)
function _apply_weight!(ZZHp, ::QualityAndProximityWeighted, tp::RandomWalkTargetPrecalculations)
    sum(ZZHp .*= tp.M)
end
function _apply_weight!(ZZHp, ::QualityWeighted, tp::RandomWalkTargetPrecalculations)
    tp.qˢ' * ZZHp * tp.qᵗ
end
function _apply_weight!(ZZHp, ::ProximityWeighted, tp::RandomWalkTargetPrecalculations)
    sum(ZZHp .*= tp.K)
end

# RandomShortestPath
function compute(::EdgeBetweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    (; Z, Zⁱ, Zrows, IW_adj_factorization, qˢ, qᵗ) = tp
    workspace1, workspace2 = workspaces(tp)
    qˢZⁱqᵗ = workspace1 .= qˢ .* Zⁱ .* qᵗ
    # QZⁱᵀZ = qˢZⁱqᵗ' / A
    QZⁱᵀZ = ldiv!(tp, IW_adj_factorization, qˢZⁱqᵗ)'
    RHS = workspace2 .= QZⁱᵀZ .- sum(qˢ) .* qᵗ .* Zⁱ[target.node, 1] .* Zrows 
    return _combine_edge_betweenness(W, Z, RHS, target)
end
function compute(::EdgeBetweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    (; Z, Zⁱ, MZⁱ, Zrows, IW_adj_factorization, workspace) = tp
    MᵀZ = ldiv!(tp, IW_adj_factorization, MZⁱ)' # MᵀZ = MZⁱ' / A
    RHS = workspace .= MᵀZ .- sum(MZⁱ) * Zⁱ[target(tp).node, 1] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target)
end
function compute(::Betweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    (; Z, Zⁱ, qˢ, qᵗ, IW_adj_factorization, workspace) = tp
    qˢZⁱqᵗ = workspace .= qˢ .* Zⁱ .* qᵗ
    # TODO: explain why this is needed
    qˢZⁱqᵗ[target(tp).node, 1] -= sum(qˢ) * qᵗ * Zⁱ[target(tp).node, 1]
    ZqˢZⁱqᵗZt = ldiv!(tp, IW_adj_factorization, qˢZⁱqᵗ) .*= Z

    return sum(ZqˢZⁱqᵗZt)
end
function compute(::Betweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    (; Z, M, MZⁱ, Zⁱ, IW_adj_factorization, workspace) = tp
    # Find the scaling factor:
    # If any of the values of MZⁱ is above one then there is a risk of overflow,
    # so we scale the matrix and apply a scale factor
    λ = max(1.0, maximum(MZⁱ))
    MZⁱλ = workspace .= MZⁱ .*= inv(λ)
    # TODO: comment what is this for
    MZⁱλ[target(tp).node, 1] = sum(MZⁱλ) * Zⁱ[target(tp).node, 1]
    # Solve: ZMZⁱt = (I - W)' \ MZⁱ
    ZMZⁱt = ldiv!(tp, IW_adj_factorization, MZⁱλ) .*= Z .* λ 
    
    return sum(ZMZⁱt)
end

function _combine_edge_betweenness(W, Z, X, target)
    t = target.node
    edge_betweennesses = spzeros(size(W, 1))
    for i in axes(W, 1)
        x = W[i, t]
        if x > 0 
            edge_betweennesses[i] = x * Z[j, t] * X[i, 1]
        end
    end
end

# Sensitivity
function compute(gm::Sensitivity, tp::RandomisedShortestPathTargetPrecalculations)
    if wrt(gm) <: Union{Affinity,Cost,CostAndAffinitySensitivityContext} 
        S_e_aff, S_e_cost = _sensitivity(connectivity_measure(gm), tp)

        if unitless 
            _scale!(S_e_aff, wrt(gm), tp)
            _scale!(S_e_cost, wrt(gm), tp)
        end
        target_sensitivity = if wrt(gm) <: Affinity
            sum(S_e_aff)
        elseif wrt(gm) <: Cost
            sum(S_e_cost)
        elseif wrt(gm) <: AffinityAndCost
            diff_C_A = ConScape.mapnz(diff_C_A_fun, affinitymatrix(g))
            S_e_total = S_e_aff .+ S_e_cost .* diff_C_A
            sum(S_e_total)
        elseif wrt(gm) <: CostAndAffinity
            diff_A_C = ConScape.mapnz(diff_A_C_fun, affinitymatrix(g))
            S_e_total = S_e_aff .* diff_A_C .+ S_e_cost
            sum(S_e_total)
        end
    elseif wrt(gm) <: Qualities
        (; qˢ, qᵗ, K, workspace) = tp
        # TODO make this single-target
        target_sensitivity = workspace .= K .+ transpose(K) .*= qᵗ 
        if unitless
            target_sensitivity *= qˢ[target.node]
        end
    end

    return target_sensitivity
end

function _sensitivity(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations)
    (; A, C, W, Z, CW, IW, Zⁱ, MZⁱ, Zrows, workspace) = tp
    diff_KD = _diff_KD(distance_transformation(tp))
    # TODO convert all / \ to ldiv!

    # MZⁱ = workspace1 .= M .* Zⁱ 
    # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ
    Y = ldiv(tp, IW, mul!(workspace, CW * Z))
    C̄ᵣ = Y .* Zⁱ # Expected costs of REGULAR paths
    MᵀZ = MZⁱ' / IW

    k̂diagZⁱ = sum(MZⁱ) .* Zⁱ[target.node, 1]

    X3 = k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ .* C̄ᵣ[target.node, 1]
    X5 = ((K̂ .* C̄ᵣ)' - (K̂ᵀZ * CW) + (X3 * CW)) / IW - k̂diagC̄Zⁱ .* Zrows # "X1- X2 - X4"
    X3 .= K̂ᵀZ .- X3

    kΣ = copy(W) # k-weighted negative covariance matrix
    kB = copy(W) # k-weighted edge betweenness matrix

    for i in axes(W, 1)
        w = W[i, 1]
        w > 0 || continue
        kB[i, 1] *= (Z[j, 1]' * X3[1, i])[1]
        kΣ[i, 1] *= (Z[j, 1]' * X5[1, i])[1] - (Y[j, 1]' * X3[1, i])[1] - C[i, 1] * kB[i, 1] / w
    end

    kΣ_node = sum(kΣ, dims=2)
    Ae = sum(A, dims=2)

    S_cost = kB + θ * kΣ

    Idx = W .> 0
    Aⁱ = mapnz(x -> inv(x), A)
    S_aff = (kΣ_node ./ Ae) .* Idx - kΣ .* Aⁱ

    return S_aff, S_cost
end
function _sensitivity(::PowerMeanProximity, tp::RandomisedShortestPathTargetPrecalculations)
    rowsums = sum(affinity(tp), dims=2)
    bet_edge_k = compute(EdgeBetweenness{QualityAndProximityWeighted}(), tp)
    bet_node_k = compute(Betweenness{QualityAndProximityWeighted}(), tp)

    Idx = A .> 0
    Aⁱ = mapnz(inv, A)

    S_e_cost = -bet_edge_k
    S_e_aff = (bet_edge_k .* Aⁱ .- (bet_node_k ./ rowsums) .* Idx) .* θ

    return S_aff, S_cost
end

_diff_CA_fun(::MinusLog) = x -> -inv(x)
_diff_AC_fun(::MinusLog) = x -> -(x)
_diff_CA_fun(::Inv) = x -> -inv(x^2) # TODO fix
_diff_AC_fun(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = -K .* x.α
_diff_KD(::Inv) = -K .^ 2

# TODO a more specific name
_scale!(A, ::Union{Affinity,AffinityAndCost}, g) = A .*= affinitymatrix(g)
_scale!(A, ::Union{Cost,CostAndAffinity}, g) = A .*= costmatrix(g)
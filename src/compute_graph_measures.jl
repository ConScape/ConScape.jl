#=
`compute` and `compute_target` for GraphMeasures.  eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

"""
    compute(::MovementMode, gm::GraphMeasure, g::GridPrecalculations[, target::Int)

Computes results of a single target pixel for a graph measure
and movement mode. 

`compute` is called inside `compute` or in the loop in `solve!` for 
VectorSolver/LinearSolver. 
"""
function compute end

const Targets = @NamedTuple{spatialid::CartesianIndex{2},targetid::Int,targetnode::Int}

# We specialise on returntrait
function compute(measures::NamedTuple, gp::GridPrecalculations)
    # Generate outputs for each graph measure
    outputs = map(measure -> allocate_output(measure, gp), measures)
    # Loop over targets
    for (i, t) in enumerate(targetnodes(gp))
        # Specify target indices
        target = Target(g.id_to_grid_coordinate_list[i], t, i)
        # Precalculate for this target and graph measures
        target_precalculations = init(gp, target)
        # Compute everything for this target and graph measures
        # We use a reduction so we can accumulate stored outputs as it runs
        foreach(measures, outputs) do (measure, output)
            # Compute a measure for this target
            v = compute(measure, target_precalculations)
            # Write values to output depending on returntrait
            output = _update!(output, measure, v, target)
        end
    end

    return outputs
end

# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure
_update!(output, gm, v, target) = _update!(output, returntrait(gm), v, target) 
_update!(output, ::AssignDenseSpatial, v::Number, target) = output[target.spatialid] = v
_update!(output, ::SumDenseSpatial, v::AbstractMatrix, target) = output .+= v
_update!(output, ::SumScalar, v::AbstractMatrix, target) = output + v
# Not sure this one makes sense
_update!(output, ::AssignSparse, v::AbstactVector, target) = output[:, target.targetid] = v

# LeastCost
function compute(gm::Betweenness, tp::LeastCostTargetPrecalculations)
    # Calculate distances
    targetid = tp.target.targetid
    (; cost_weighted_digraph, workspace) = tp
    shorted_paths = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, targetid)
    shortest_paths_en = Graphs.enumerate_paths(shorted_paths)

    # And maybe transform them
    # TODO untangle this logic so apply_weight handles everythign
    k = if isnothing(distance_transformation(tp))
        workspace1 .= 1.0
    else
        workspace1 .= distance_transformation(cm).(shorted_paths.dists)
    end
    apply_weight!(k, gm, tp)

    # TODO what does all this do...
    shortest_paths_en[targetid] = [targetid]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)

    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw * k
end

# RandomWalk 
function compute(::EdgeBetweenness{Weighting}, tp::RandomWalkTargetPrecalculations) where Weighting
    nodebet = compute(Betweenness(Weighting()), tp)
    return nodebet * pref[t.spatialid]
end
function compute(::Betweenness{QualityWeighted}, tp::RandomWalkTargetPrecalculations)
    Z, H, qˢ, qᵗ, p, workspace = tp.fundamental, tp.hitting_time, tp.source_quality, tp.target_quality, tp.stationary_distribution, tp.workspaces[1]
    return qˢ' * _betweenness!(workspace, Z, H, p, t) * qᵗ
end
function compute(::Betweenness{QualityAndProximityWeighted}, tp::RandomWalkTargetPrecalculations)
    Z, H, K, p, workspace = tp.probability, tp.cost, tp.fundamental, tp.hitting_time, tp.quality_weighted_proximity, tp.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end
function compute(::Betweenness{ProximityWeighted}, tp::RandomWalkTargetPrecalculations)
    Z, H, K, p, workspace = tp.fundamental, tp.hitting_time, tp.proximity, tp.stationary_distribution, tp.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end

# TODO this only works for simmetrical Z
_betweenness!(workspace, Z, H, p, t::Targets) =
    workspace .= view(Z, :, 1) .- view(Z, :, t)' .+ H .* p[t]

apply_weight!(k, ::Unweigthed, tp::TargetPrecalculations) = k
apply_weight!(k, ::QualityAndProximityWeighted, tp::TargetPrecalculations) =
    k .*= source_qualities(tp) .* target_qualities(pt)[tp.targets.targetnode]

# RandomShortestPath

# Betweenness: TODO move code here from RSP and rewrite per-target
function compute(::Betweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    targetnode = tp.target.targetnode
    (; Z, qˢ, Zⁱ) = tp
    workspace1, workspace2 = tp.workspaces
    qᵗ = g.target_qualities[targetnode]

    qˢZⁱqᵗ = workspace1 .= qˢ .* Zⁱ .* qᵗ

    # TODO: explain why this is needed
    qˢZⁱqᵗ[targetnode, 1] -= sum(qˢ) * qᵗ * Zⁱ[targetnode, 1]

    ZqˢZⁱqᵗZt = ldiv!(solver(tp), Aadj_init, qˢZⁱqᵗ; B_copy=copy!(workspace2, qˢZⁱqᵗ))
    ZqˢZⁱqᵗZt .*= Z

    return ZqˢZⁱqᵗZt
end
function compute(::EdgeBetweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    # Zrows = ldiv!(solver(tp), IWadj_factorization, b; B_copy=copy!(workspace2, B))'
    (; Z, Zⁱ, Zrows, qˢ) = tp
    workspace1, workspace2 = tp.workspaces
    qᵗ = g.target_qualities[target.targetnode]

    qˢZⁱqᵗ = workspace1 .= qˢ .* Zⁱ .* qᵗ
    # QZⁱᵀZ = qˢZⁱqᵗ' / A
    QZⁱᵀZ = ldiv!(solver(tp), Aadj_factorization, qˢZⁱqᵗ; B_copy=copy!(workspace2, qˢZⁱqᵗ))'

    RHS = workspace3 .= QZⁱᵀZ .- Zrows .* sum(qˢ) .* qᵗ .* Zⁱ[t.targetnode, 1]
    return _combine_edge_betweenness(W, Z, RHS, target)
end
function compute(::EdgeBetweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    (; MZⁱ, Zrows) = tp # MZⁱ = workspace1 .= M .*= Zⁱ
    workspace1, workspace2 = tp.workspaces
    # MᵀZ = MZⁱ' / A
    MᵀZ = ldiv!(solver(tp), IWadj_init, MZⁱ; copy=copy!(workspace1, MZⁱ))'
    k̂diagZⁱ = sum(MZⁱ) * Zⁱ[target.targetnode, 1]
    MᵀZ_minus_diag = workspace2' .=  MᵀZ .- k̂diagZⁱ .* Zrows'

    return _combine_edge_betweenness(W, Z, MᵀZ_minus_diag, target)
end
function compute(::Betweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    # MZⁱ = workspace2 .= M .* Zⁱ
    (; Z, MZⁱ, Zⁱ) = tp
    workspace1 = tp.workspaces

    # Divide the  by Z
    # Find the scaling factor:
    # If any of the values of KZⁱ is above one then there is a risk of overflow,
    # so we scale the matrix and apply a scale factor
    λ = max(1.0, maximum(MZⁱ))
    MZⁱλ = let workwspace .= MZⁱ
    # TODO: comment what is this for
    MZⁱλ[target.targetnode, 1] -= sum(MZⁱ) / λ * Zⁱ[target.targetnode, 1]
    # Normalise before the solve
    MZⁱλ ./= λ
    # Solve: ZKZⁱt = (I - W)' \ KZⁱ
    B_copy = copy!(workspace1, MZⁱλ)
    ZMZⁱt = ldiv!(solver(tp), IWadj_init, MZⁱλ; B_copy) .* λ .* Z

    return sum(ZMZⁱt)
end

function _combine_edge_betweenness(W, Z, X, target)
    t = target.targetnode
    edge_betweennesses = spzeros(size(W, 1))
    for i in axes(W, 1)
        x = W[i, t]
        if x > 0 
            edge_betweennesses[i] = x * Z[j, t] * X[i, 1]
        end
    end
end

# ConnectedHabitat 
compute(::ConnectedHabitat, tp::TargetPrecalculations) = tp.landscape_matrix

function compute(gm::Sensitivity, tp::RandomisedShortestPathTargetPrecalculations)
    target_sensitivity = if gm.context <: Union{Affinity,Cost,CostAndAffinitySensitivityContext} 
        S_e_aff, S_e_cost = sensitivity(cm, tp)

        if unitless 
            _scale!(S_e_aff, gm.context, tp)
            _scale!(S_e_cost, gm.context, tp)
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
        K = tp.proximiites
        KK = workspace2 .= K .+= transpose(K)
        target_sensitivity = mul!(workspace3, K, g.target_qualities[target.spatialid])
        if unitless
            target_sensitivity .* g.source_qualities[target.spatialid]
        end
        target_sensitivity
    end

    target_sensitivity * isnan(g.source_qualities[target.spatialid]) ? NaN : 1.0
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

function sensitivity(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations)

    A::SparseMatrixCSC,
    C::SparseMatrixCSC,
    θ::Real,
    W::SparseMatrixCSC,
    Z::AbstractMatrix,
    CW::SparseMatrixCSC,
    IW::SparseMatrixCSC,
    Zⁱ::AbstractMatrix,
    MZⁱ::AbstractMatrix,
    Zrows

        diff_K_D = _diff_KD(distance_transformation)
    # TODO convert all / \ to ldiv!

    # MZⁱ = workspace1 .= M .* Zⁱ 
    # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ
    Y = IW \ (CW * Z)
    C̄ᵣ = Y .* Zⁱ # Expected costs of REGULAR paths
    MᵀZ = MZⁱ' / IW

    k̂diagZⁱ = sum(MZⁱ) .* Zⁱ[target.targetnode, 1]

    X3 = k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ .* C̄ᵣ[target.targetnode, 1]
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

    S_cost = kB + θ * kΣ#/θ

    Idx = W .> 0
    Aⁱ = mapnz(x -> inv(x), A)
    S_aff = (kΣ_node ./ Ae) .* Idx - kΣ .* Aⁱ

    return S_aff, S_cost
end

function sensitivity(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations)
    rowsums = sum(affinity(tp), dims=2)
    bet_edge_k = compute(EdgeBetweenness{QualityAndProximityWeighted}(), tp)
    bet_node_k = compute(Betweenness{QualityAndProximityWeighted}(), tp)

    Idx = A .> 0
    Aⁱ = mapnz(inv, A)

    S_e_cost = -bet_edge_k
    S_e_aff = (bet_edge_k .* Aⁱ .- (bet_node_k ./ rowsums) .* Idx) .* θ

    return S_aff, S_cost
end
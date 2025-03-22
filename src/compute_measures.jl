#=
`compute` and `compute_target` for GraphMeasures.  eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

# TODO: the loop here doesn't decompose to single targets, so the full Z matrix seems to be needed.
# this is a problem for memory use in e.g. BatchProblem, D may be large fraction of 
# the available memory per node on the cluster (~3gb per core)

function compute(
    ::Union{ExpectedCost,FreeEnergyDistance}, gi::GridInit{<:RandomWalk}
)
    (; P, C) = gi
    PC = sum(P .* C; dims=2)
    IP = I - P
    n = LinearAlgebra.checksquare(P)
    D = Array{eltype(P)}(undef, n, n)
    v = zeros(n)
    vt = 0
    # TODO: this may be a problem for VectorSolver and memory reduction
    for target in 1:n
        if target > 1
            # Here we update IP and PC with the previous target
            IP[target-1, :] = v
            PC[target-1] = vt
        end
        vt = PC[target]
        v .= view(IP, target, :)
        PC[target] = 0
        IP[target, :] .= zero(eltype(P))
        IP[target, target] = 1
        D[:, target] = IP \ PC
    end
    # Returns large dense matrix
    return D
end
function compute(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Z, Zⁱ, CW, IW_factorization, workspace) = ti
    # Solve: IW \ ((C .* W) * Z)
    b = mul!(workspace, CW, Z)
    C̄ = ldiv!(ti, IW_factorization, b) .*= Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[target(ti).node, 1]
    return C̄
end
function compute(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    θ = theta(ti)
    (; survival_probabilities, workspace) = ti
    return workspace .= -log.(max.(0, survival_probabilities)) ./ θ
end
function compute(::PowerMeanProximity, ti::TargetInit{<:RSP})
    θ = theta(ti)
    (; survival_probabilities, workspace) = ti
    return workspace .= survival_probabilities .^ (1 / θ)
end
function compute(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z, workspace) = ti
    return workspace .= Z ./ Z[target(ti).node, 1]
end
# Mean Kullback-Leibler Divergence
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:LeastCost})
    (; Pref, cost_weighted_digraph, qˢ, qᵗ) = ti
    from, to, output = workspaces(ti)

    # Calculate shortest paths
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target(ti).node)
    parents = dsp.parents
    parents[target(ti).node] = target(ti).node

    # Initialise arrays
    fill!(output, 0)
    from .= 1:n
    to .= parents

    # TODO explain what this loop does
    while true
        notdone = false
        for i in 1:n
            fromᵢ, toᵢ = from[i], to[i]
            notdone |= (fromᵢ != toᵢ)
            fromᵢ == toᵢ && continue
            output[i] += -log(Pref[fromᵢ, toᵢ])
            from[i] = parents[toᵢ]
        end
        if !notdone
            break
        end
        from, to = to, from
    end
    return sum(output .*= qˢ) * qᵗ # qs' * output * qt
end
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:RandomWalk})
    # Trivially returns zero ?
    return 0.0
end
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    θ = theta(ti)
    (; free_energy_distances, expected_costs, qˢ, qᵗ, workspace) = ti
    diff = workspace .= free_energy_distances .- expected_costs
    # qs' * diff * qt * θ
    return sum(diff .*= qˢ) * qᵗ * θ
end

# What are these, how are they different to the RSP versions?
# compute(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

# ConnectedHabitat 
compute(::ConnectedHabitat, ti::TargetInit) = ti.M

# Betweenness

# LeastCost
function compute(gm::Betweenness, ti::TargetInit{<:LeastCost})
    (; shortest_paths_en, dijk) = ti
    # TODO what does all this do...
    shortest_paths_en[targetid] = [targetid]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)
    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw .*= _weight(weighting(gm), ti)
end

# RandomWalk 
function compute(
    ::EdgeBetweenness{Weighting}, ti::TargetInit{<:RandomWalk}
) where Weighting
    return compute(Betweenness(Weighting()), ti) * pref[target(ti).spatial]
end
function compute(bet::Betweenness, ti::TargetInit{<:RandomWalk})
    (; Z1, Z, H, p, workspace) = ti
    return workspace .= Z1 .- Z .+ H .* p[target(ti).node] .* _weight(weighting(gm), ti)
end

# RandomShortestPath
function compute(::EdgeBetweenness{QualityWeighted}, ti::TargetInit{<:RSP})
    (; Z, Zⁱ, Zrows, QZⁱ, W, IW_adj_factorization, qˢ, qᵗ, workspace) = ti
    # QZⁱᵀZ = qˢZⁱqᵗ' / A
    QZⁱc = workspace .= QZⁱ
    QZⁱᵀZ = ldiv!(ti, IW_adj_factorization, QZⁱc)
    RHS = workspace .= QZⁱᵀZ .- sum(qˢ) .* qᵗ .* Zⁱ[target(ti).node, 1] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target(ti))
end
function compute(
    ::EdgeBetweenness{QualityAndProximityWeighted}, ti::TargetInit{<:RSP}
)
    (; W, Z, Zⁱ, M, Zrows, IW_adj_factorization, workspace) = ti
    MZⁱ = workspace .= M .* Zⁱ
    k = sum(MZⁱ)
    MᵀZ = ldiv!(ti, IW_adj_factorization, MZⁱ) # MᵀZ = MZⁱ' / A
    RHS = workspace .= MᵀZ .- k * Zⁱ[target(ti).node, 1] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target(ti))
end
function compute(m::Betweenness, ti::TargetInit{<:RSP})
    (; Z, Zⁱ, IW_adj_factorization, workspace) = ti
    # Find the scaling factor:
    # If any of the values of MZⁱ is above one then there is a risk of overflow,
    X = _weight(weighting(m), ti)
    λ = max(1.0, maximum(X))
    XZⁱt = workspace .= X .* Zⁱ
    # TODO: explain what this subtraction does
    XZⁱt[target(ti).node] -= Zⁱ[target(ti).node] * sum(X)
    # Scale MZⁱ with λ
    XZⁱtλ = XZⁱt .*= inv(λ)
    # Solve (I - W)' \ MZⁱλ, then multiply by Z and λ scaling
    return ldiv!(ti, IW_adj_factorization, XZⁱtλ) .*= λ .* Z
end

_weight(::Unweighted, ti::TargetInit) = 1
_weight(::ProximityWeighted, ti::TargetInit) = ti.K
_weight(::QualityAndProximityWeighted, ti::TargetInit) = ti.M
_weight(::QualityWeighted, ti::TargetInit) = ti.Q

function _combine_edge_betweenness(W, Z, X, t::TargetID)
    edge_betweennesses = spzeros(size(W, 1))
    for i in axes(W, 1)
        w = W[i, t.node]
        if w > 0 
            edge_betweennesses[i] = w * Z[t.node] * X[i]
        end
    end
    return edge_betweennesses
end

# Sensitivity
function compute(gm::Sensitivity, ti::TargetInit{<:RSP})
    # TODO calculate this in GridInit
    # diff_C_A = ConScape.mapnz(_diff_C_A_fun(ti), affinitymatrix(ti))
    # diff_A_C = ConScape.mapnz(_diff_A_C_fun(ti), affinitymatrix(ti))

    if wrt(gm) isa Union{Affinity,Cost,CostAndAffinitySensitivityContext} 
        (; diff_C_A, diff_A_C) = ti
        S_e_aff, S_e_cost = _sensitivity(proximity_measure(ti), ti)

        if unitless 
            _scale_uniless!(S_e_aff, wrt(gm), ti)
            _scale_uniless!(S_e_cost, wrt(gm), ti)
        end
        target_sensitivity = if wrt(gm) isa Affinity
            S_e_aff
        elseif wrt(gm) isa Cost
            S_e_cost
        elseif wrt(gm) isa AffinityAndCost
            S_e_total = S_e_aff .+ S_e_cost .* diff_C_A
            S_e_total
        elseif wrt(gm) isa CostAndAffinity
            S_e_total = S_e_aff .* diff_A_C .+ S_e_cost
            S_e_total
        end
    elseif wrt(gm) isa Qualities
        (; qˢ, qᵗ, K, unitless, workspace) = ti
        # TODO make this single-target
        # Need a summed source proximities vector
        # Also split in respect to source and target quality
        target_sensitivity = workspace .= K .+ transpose(K) .*= qᵗ 
        if unitless
            target_sensitivity *= qˢ[target.node]
        end
    end

    return target_sensitivity
end

function _sensitivity(::ExpectedCost, ti::TargetInit{<:RSP})
    (; A, C, W, K, Z, CW, IW, Zⁱ, MZⁱ, Zrows, workspace) = ti
    diff_KD = _diff_KD(K, distance_transformation(ti))
    # TODO convert all / \ to ldiv!

    # MZⁱ = workspace1 .= M .* Zⁱ 
    # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ
    Y = ldiv(ti, IW, mul!(workspace, CW * Z))
    C̄ᵣ = Y .* Zⁱ # Expected costs of REGULAR paths
    MᵀZ = MZⁱ' / IW

    k̂diagZⁱ = sum(MZⁱ) .* Zⁱ[target.node, 1]

    X3 = k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ .* C̄ᵣ[target.node, 1]
    X5 = ((K̂ .* C̄ᵣ)' - (MᵀZ * CW) + (X3 * CW)) / IW - k̂diagC̄Zⁱ .* Zrows # "X1- X2 - X4"
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
function _sensitivity(::PowerMeanProximity, ti::TargetInit{<:RSP})
    # Aⁱ = mapnz(inv, A)
    # rowsums = sum(affinity(ti), dims=2)
    (; A, Aⁱ, rowsums) = ti
    id = target(ti).id

    bet_edge_k = compute(EdgeBetweenness{QualityAndProximityWeighted}(), ti)
    bet_node_k = compute(Betweenness{QualityAndProximityWeighted}(), ti)

    S_aff = workspace .= (bet_edge_k .* Aⁱ[:, id] .* A[:, id] .> 0 .- (bet_node_k ./ rowsums)) .* θ
    S_cost = workspace .= .-(bet_edge_k)

    return S_aff, S_cost
end

_diff_CA_fun(::MinusLog) = x -> -inv(x)
_diff_AC_fun(::MinusLog) = x -> -(x)
_diff_CA_fun(::Inv) = x -> -inv(x^2) # TODO fix
_diff_AC_fun(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = -K .* x.α
_diff_KD(::Inv) = -K .^ 2

_scale_uniless!(A, ::Union{Affinity,AffinityAndCost}, g) = A .*= affinitymatrix(g)
_scale_uniless!(A, ::Union{Cost,CostAndAffinity}, g) = A .*= costmatrix(g)
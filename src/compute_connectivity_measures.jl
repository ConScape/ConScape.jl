# TODO: the loop here doesn't decompose to single targets, so the full Z matrix seems to be needed.
# this is a problem for memory use in e.g. BatchProblem, D may be large fraction of 
# the available memory per node on the cluster (~3gb per core)
function compute(::HittingTime, gp::RandomWalkGridPrecalculations)
    (; P, C) = gp
    PC = sum(P .* C; dims=2)
    IP = I - P
    # TODO does this have to be square?
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
function compute(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations)
    (; Z, Zⁱ, CW, IW_factorization, workspace) = tp
    # Solve: IW \ ((C .* W) * Z)
    b = mul!(workspace, CW, Z)
    C̄ = ldiv!(tp, IW_factorization, b) .*= Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[target(tp).node, 1]
    return C̄
end
function compute(::FreeEnergyDistance, tp::RandomisedShortestPathTargetPrecalculations)
    θ = theta(tp)
    (; survival_probabilities, workspace) = tp
    return workspace .= -log.(max.(0, survival_probabilities)) ./ θ
end
function compute(::PowerMeanProximity, tp::RandomisedShortestPathTargetPrecalculations)
    θ = theta(tp)
    (; survival_probabilities, workspace) = tp
    return workspace .= survival_probabilities .^ (1 / θ)
end
function compute(::SurvivalProbability, tp::RandomisedShortestPathTargetPrecalculations) 
    (; Z, workspace) = tp
    return workspace .= Z ./ Z[target(tp).node, 1]
end
# Mean Kullback-Leibler Divergence
function compute(::KullbackLeiblerDivergence, tp::LeastCostTargetPrecalculations)
    (; Pref, cost_weighted_digraph, qˢ, qᵗ) = tp
    from, to, output = workspaces(tp)

    # Calculate shortest paths
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target(tp).node)
    parents = dsp.parents
    parents[target(tp).node] = target(tp).node

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
            v = Pref[fromᵢ, toᵢ]
            output[i] += -log(v)
            from[i] = parents[toᵢ]
        end
        if !notdone
            break
        end
        from, to = to, from
    end
    return sum(output .*= qˢ) * qᵗ # qs' * output * qt
end
function compute(::KullbackLeiblerDivergence, tp::RandomWalkTargetPrecalculations)
    # Trivially returns zero ?
    return 0.0
end
function compute(::KullbackLeiblerDivergence, tp::RandomisedShortestPathTargetPrecalculations)
    θ = theta(tp)
    (; free_energy_distances, expected_costs, qˢ, qᵗ, workspace) = tp
    diff = workspace .= free_energy_distances .- expected_costs
    # qs' * diff * qt * θ
    return sum(diff .*= qˢ) * qᵗ * θ
end

# What are these, how are they different to the RSP versions?
# compute(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations) = first(bellman_ford(tp))
# compute(::FreeEnergyDistance, tp::RandomisedShortestPathTargetPrecalculations) = last(bellman_ford(tp))

bellman_ford(tp::RandomisedShortestPathTargetPrecalculations) =
    first(bellman_ford(probabilitymatrix(tp), costmatrix(tp), theta(tp), target_id(tp), approx(tp)))
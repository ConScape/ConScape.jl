# TODO: the loop here doesn't decompose to single targets, so the full Z matrix seems to be needed.
# this is a problem for memory use in e.g. BatchProblem, D may be large fraction of 
# the available memory per node on the cluster (~3gb per core)
function compute(::HittingTime, g::RandomWalkGridPrecalculations)
    (; P, C) = g
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

function compute(
    ::ExpectedCost, 
    gp::RandomisedShortestPathTargetPrecalculations, 
)
    (; CW, Z, IW_init, workspaces) = gp
    workspace1, workspace2 = workspaces
    # Solve: IW \ (C .* W * Z)
    B = mul!(workspace1, CW, Z)
    C̄ = ldiv!(solver, IW_init, B; B_copy=copy!(workspace2, B))
    # TODO comment why we divide by Z
    C̄ ./= Z
    # Clean up NaNs
    replace!(C̄, NaN => Inf)
    # Subtract the cost at the target
    C̄ .-= C̄[target.node, 1]
    return C̄
end
function compute(cm::FreeEnergyDistance, gp::RandomisedShortestPathTargetPrecalculations)
    θ = movement(cm).θ
    (; survival_probability, workspaces) = gp
    fed = pop!(workspaces) 
    return fed .= -log.(max.(zero(eltype(Z)), survival_probability)) ./ θ
end
function compute(::PowerMeanProximity, gp::RandomisedShortestPathTargetPrecalculations)
    θ = movement(gp).θ
    (; survival_probability) = gp
    pmp = pop!(gp.workspaces)
    return pmp .= survival_probability .^ (1 / θ)
end
function compute(::SurvivalProbability, gp::RandomisedShortestPathTargetPrecalculations) 
    sp = pop!(gp.workspaces)
    return sp .= Z ./ Z[target, 1]
end

# Mean Kullback-Leibler Divergence
function compute(::KullbackLeiblerDivergence, gp::LeastCostTargetPrecalculations)
    (; Pref, cost_weighted_digraph, target) = gp
    from, to, output = gp.workspaces

    # Calculate shortest paths
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target.node)
    parents = dsp.parents
    parents[target.node] = target.node

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
    # qs' * output * qt
    return sum(output1 .*= g.qs) * g.qt[target.node]
end
function compute(::KullbackLeiblerDivergence, gp::RandomWalkTargetPrecalculations)
    # Trivially returns zero ?
    return 0.0
end
function compute(cm::KullbackLeiblerDivergence, gp::RandomisedShortestPathTargetPrecalculations)
    g = grid(gp)
    θ = movement(cm).θ
    (; target, free_energy_distances, expected_costs, workspaces) = gp
    diff = workspace .= free_energy_distances .- expected_costs
    # qs' * diff * qt * θ
    return sum(diff .*= g.qs) * g.qt[target.node] * θ
end

# What are these, how are they different to the RSP versions?
# compute(::ExpectedCost, tp::RandomisedShortestPathTargetPrecalculations) = first(bellman_ford(gp))
# compute(::FreeEnergyDistance, tp::RandomisedShortestPathTargetPrecalculations) = last(bellman_ford(gp))

bellman_ford(tp::RandomisedShortestPathTargetPrecalculations) =
    first(bellman_ford(probabilitymatrix(tp), costmatrix(tp), theta(tp), target_id(tp), approx(tp)))
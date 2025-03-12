# TODO: the loop here doesn't decompose to single targets, so the full Z matrix seems to be needed.
# this is a problem for memory use in e.g. BatchProblem, D may be large fraction of 
# the available memory per node on the cluster (~3gb per core)
function compute(::DistanceMetric{<:RandomWalk}, g::Grid)
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
    ::ExpectedCost{<:RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations, 
    targets::AbstractVector,
)
    (; W, C, CW, Z, A, A_init, workspace1, workspace2) = gp
    B = mul!(workspace1, CW, Z)
    C̄ = ldiv!(solver, A_init, B; B_copy=copy!(workspace2, B))
    C̄ ./= Z
    replace!(C̄, NaN => Inf)
    dˢ = view(workspace2, 1, :)
    for j in axes(Z, 2)
        dˢ[j] = C̄[targets[j], j]
    end
    C̄ .-= dˢ'
    return copy(C̄)
end
function compute(
    cm::FreeEnergyDistance{<:::RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations, 
    targets,
)
    θ = movement(cm).θ
    (; survival_probability, free_energy_distances_buffer) = gp
    free_energy_distances .= -log.(max.(zero(eltype(Z)), survival_probability)) ./ θ
    return free_energy_distances
end
function compute(
    ::SurvivalProbability{<:RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations, 
    targets::AbstractVector
) 
    Z .* inv.((Z[i, j] for (j, i) in enumerate(targets)))'
end
function compute(
    ::SurvivalProbability{<:RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations, 
    target::Int
) 
    Z ./ Z[target, 1]
end
function compute(
    cm::PowerMeanProximity{<:::RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations, 
    targets::AbstractVector, 
)
    θ = movement(cm).θ
    (; survival_probability) = gp
    return survival_probability .^ (1 / θ)
end

function mean_kl_divergence(grsp::Union{GridRSP,NamedTuple}, free_energy_distances, expected_costs;
    workspaces=(similar(grsp.Z),), kw...
)
    g = grsp.g
    fed_exp = workspaces[1] .= free_energy_distances .- expected_costs
    return g.qs' * fed_exp * g.qt * grsp.θ
end

function compute(
    ::KullbackLeiblerDivergence{<:LeastCost}, 
    gp::LeastCostPrecalculations;
)
    workspace1 = workspaces[1]
    g = grsp.g
    C = g.costmatrix
    cost_weighted_digraph = SimpleWeightedDiGraph(C)
    n = size(C, 1)
    from = Array{Int}(undef, n)
    kl_div = Array{Float64}(undef, n)
    # Previously
    # div = hcat([least_cost_kl_divergence(C, grsp.Pref, i; cost_weighted_digraph, from, kl_div, kw...) for i in g.targetnodes]...)
    div = workspace1
    for i in g.targetnodes
        div[i, :] .= least_cost_kl_divergence(C, grsp.Pref, i; cost_weighted_digraph, from, kl_div, kw...)
    end
    # Why is it q wighted here but not per target?
    return g.qs' * div * g.qt
end
function compute(
    ::KullbackLeiblerDivergence{<:LeastCost}, 
    gp::LeastCostPrecalculations,
    target::Integer;
)
    (; C, Pref, cost_weighted_digraph, dsp) = gp
    n = size(C, 1)
    from = Array{Int}(undef, n)
    output = Array{Float64}(undef, n)

    # Calculate shortest paths
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target)
    parents = dsp.parents
    parents[targetnode] = targetnode

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
            if fromᵢ == toᵢ
                continue
            end
            v = Pref[fromᵢ, toᵢ]
            output[i] += -log(v)
            from[i] = parents[toᵢ]
        end
        if !notdone
            break
        end
        from, to = to, from # Pointer swap (yes but why?)
    end
    return output
end
function compute(
    ::KullbackLeiblerDivergence{<:RandomWalk}, 
    gp::LeastCostPrecalculations,
    target::Integer;
)
    # Trivially returns zero
    return 0.0
end
function compute(
    cm::KullbackLeiblerDivergence{<:RandomShortestPath}, 
    gp::RandomisedShortestPathPrecalculations,
)
    g = gp.g
    θ = movement(cm).θ
    (; free_energy_distances, expected_costs, workspace) = gp
    fed_exp = workspace .= free_energy_distances .- expected_costs
    # Returns a scalar
    return g.qs' * fed_exp * g.qt * θ
end

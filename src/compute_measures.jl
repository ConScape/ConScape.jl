
function get_or_compute!(m::Measure, ti::TargetInit)
    get!(storage(ti), Symbol(m)) do
        compute(m, ti)
    end
end

######################################################################################
# Proximities

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
    # Solve: (I - W) \ (C .* W) * Z ./ Z
    C̄ = ldiv!(ti, IW_factorization, mul!(workspace, CW, Z)) .*= Zⁱ
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
    return workspace .= Z ./ Z[target(ti).node]
end


######################################################################################
# Mean Kullback-Leibler Divergence

function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:LeastCost})
    (; probability, cost_weighted_digraph, qˢ, qᵗ) = ti
    output = ti.workspace
    from = Vector{Int}(undef, length(output))
    to = Vector{Int}(undef, length(output))

    n = length(from)
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target(ti).node)
    parents = dsp.parents
    parents[target(ti).node] = target(ti).node

    # Initialise arrays
    fill!(output, 0.0)
    from .= 1:n
    to .= parents

    # TODO explain what this loop does
    while true
        notdone = false
        for i in 1:n
            fromᵢ, toᵢ = from[i], to[i]
            notdone |= (fromᵢ != toᵢ)
            fromᵢ == toᵢ && continue
            output[i] += -log(probability[fromᵢ, toᵢ])
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
    return sum(diff .*= qˢ) * qᵗ * θ # qˢ' * diff * qᵗ * θ
end

# What are these, how are they different to the RSP versions?
# compute(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

######################################################################################
# ConnectedHabitat 

compute(::ConnectedHabitat, ti::TargetInit) = ti.M

######################################################################################
# Betweenness

# LeastCost
function compute(m::Betweenness, ti::TargetInit{<:LeastCost})
    (; shortest_paths, path_allocs, workspace) = ti
    shortest_paths_enumerated = Graphs.enumerate_paths!(path_allocs, shortest_paths, 1:length(path_allocs))
    # Set the target path to only contain itself
    targetpath = resize!(shortest_paths_enumerated[target(ti).node], 1)
    targetpath[1] = target(ti).node
    # Get the target weights
    weights = _weight(m, ti)
    btw = workspace .= 0.0

    @inbounds for s in eachindex(source_ids(ti))
        w = weights[s]
        for p in shortest_paths_enumerated[s]
            btw[p] += w 
        end
    end
    return btw
end
# RandomWalk 
function compute(m::Betweenness, ti::TargetInit{<:RandomWalk})
    (; Z1, Z, H, p, workspace) = ti
    return workspace .= Z1 .- Z .+ H .* p[target(ti).node] .* _weight(m, ti)
end
# RandomShortestPath
function compute(m::Betweenness, ti::TargetInit{<:RSP})
    (; Z, Zⁱ, IW_adj_factorization, workspace) = ti
    X = _weight(m, ti)
    XZⁱt = workspace .= X .* Zⁱ
    # Find the scaling factor: if any of XZⁱ is above 1.0 there is a risk of Inf overflow
    λ = max(1.0, maximum(XZⁱt))
    # TODO: explain what this subtraction does
    XZⁱt[target(ti).node] -= Zⁱ[target(ti).node] * sum(X)
    # Scale MZⁱ with λ
    XZⁱtλ = XZⁱt .*= inv(λ)
    # Solve (I - W)' \ MZⁱλ, then multiply by Z and λ scaling
    return ldiv!(ti, IW_adj_factorization, XZⁱtλ) .*= λ .* Z
end

_weight(m::Betweenness, ti::TargetInit) = _weight(weighting(m), ti)
_weight(::Unweighted, ti::TargetInit) = 1
_weight(::ProximityWeighted, ti::TargetInit) = ti.K
_weight(::QualityAndProximityWeighted, ti::TargetInit) = ti.M
_weight(::QualityWeighted, ti::TargetInit) = ti.Q

######################################################################################
# EdgeBetweenness

# LeastCost

# TODO: implement

# RandomWalk
function compute(eb::EdgeBetweenness, ti::TargetInit{<:RandomWalk})
    return compute(Betweenness(weighting(eb)), ti) * pref[target(ti).spatial]
end

# RandomShortestPath
function compute(::EdgeBetweenness{QualityWeighted}, ti::TargetInit{<:RSP})
    (; Z, Zⁱ, Zrows, W, IW_adj_factorization, qˢ, qᵗ, workspace) = ti
    QZⁱ = workspace .= ti.QZⁱ
    k = sum(qˢ) .* qᵗ # TODO: is this a bug? why not sum(QZⁱ) as below
    QZⁱᵀZ = ldiv!(ti, IW_adj_factorization, QZⁱ)
    RHS = QZⁱᵀZ .-= k .* Zⁱ[target(ti).node, 1] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target(ti))
end
function compute(
    ::EdgeBetweenness{QualityAndProximityWeighted}, ti::TargetInit{<:RSP}
)
    (; W, Z, Zⁱ, M, Zrows, IW_adj_factorization, workspace) = ti
    MZⁱ = workspace .= M .* Zⁱ
    k = sum(MZⁱ)
    MᵀZ = ldiv!(ti, IW_adj_factorization, MZⁱ)
    RHS = MᵀZ .-= k .* Zⁱ[target(ti).node, 1] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target(ti))
end

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


######################################################################################
# Sensitivity
function compute(m::Sensitivity{<:Permeability}, ti::TargetInit{<:RSP})
    (; diff_C_A, diff_A_C) = ti
    S_e_aff, S_e_cost = _permeability_sensitivity(proximity_measure(ti), ti)

    if change(m) isa ProportionalChange
        _scale_uniless!(S_e_aff, context(m), ti)
        _scale_uniless!(S_e_cost, context(m), ti)
    end

    return if context(m) isa Affinity
        S_e_aff
    elseif context(m) isa Cost
        S_e_cost
    elseif context(m) isa AffinityAndCost
        S_e_aff .+ S_e_cost .* diff_C_A
    elseif context(m) isa CostAndAffinity
        S_e_aff .* diff_A_C .+ S_e_cost
    end
end
function compute(m::Sensitivity{<:Quality}, ti::TargetInit{<:RSP})
    (; qˢ, qᵗ, K, workspace) = ti
    # TODO make this single-target
    # Need a summed source proximities vector
    # Also split in respect to source and target quality
    target_sensitivity = workspace .= K .+ transpose(K) .*= qᵗ 
    if change(m) isa ProportionalChange
        target_sensitivity .*= qˢ[target(ti).node]
    end
    return target_sensitivity
end

function _permeability_sensitivity(::ExpectedCost, ti::TargetInit{<:RSP})
    (; A, Aⁱ, A_rowsums, C, W, M, Z, CW, IW_factorization, IW_adj_factorization, Zⁱ, MZⁱ, Zrows) = ti
    θ = theta(ti)
    # diff_KD = _diff_KD(K, distance_transformation(ti))

    # TODO this is basically ExpectedCost mashed with Betweenness K
    # Can we reuse those instead?

    # MZⁱ = workspace1 .= M .* Zⁱ 
    # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ
    Y = ldiv!(ti, IW_factorization, mul!(ti.workspace, CW, Z))
    C̄ = Y .*= Zⁱ # Expected costs of REGULAR paths

    # TODO is this flipped the right way
    # MᵀZ = MZⁱ' / IW
    MᵀZ = ldiv!(ti, IW_factorization, MZⁱ)

    k̂diagZⁱ = sum(MZⁱ) .* Zⁱ[target(ti).node, 1]

    X3 = ti.workspace .= k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ .* C̄[target(ti).node, 1]
    RHS = vec((M .* C̄)' .- (MᵀZ' * CW) .+ (X3' * CW))
    X5 = ldiv!(ti, IW_adj_factorization, RHS) .- k̂diagC̄Zⁱ .* Zrows # "X1- X2 - X4"
    X3 .= MᵀZ .- X3

    Wt = ti.workspace .= view(W, :, 1)
    kΣ = ti.workspace .= Wt # k-weighted negative covariance matrix
    kB = ti.workspace .= Wt # k-weighted edge betweenness matrix

    j = target(ti).node
    for i in eachindex(Wt)
        w = Wt[i]
        w > 0 || continue
        kB[i] *= Z[j] * X3[i]
        kΣ[i] *= (Z[j] * X5[i]) - (Y[j]' * X3[i])[1] - C[i, 1] * kB[i] / w
    end

    kΣ_node = sum(kΣ, dims=2)

    S_cost = kB + θ * kΣ
    S_aff = kΣ_node ./ A_rowsums .* (Wt .> 0) .- kΣ .* view(Aⁱ, :, target(ti).node)

    return S_aff, S_cost
end
function _permeability_sensitivity(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; A, Aⁱ, A_rowsums, workspace) = ti
    θ = theta(ti)
    id = target(ti).id

    bet_edge_k = get_or_compute!(EdgeBetweenness(QualityAndProximityWeighted()), ti)
    bet_node_k = get_or_compute!(Betweenness(QualityAndProximityWeighted()), ti)

    S_cost = workspace .= .-(bet_edge_k)
    S_aff = workspace .= (bet_edge_k .* view(Aⁱ, :, id) .* (view(A, :, id) .> 0) .- (bet_node_k ./ A_rowsums)) .* θ

    return S_aff, S_cost

    # survival_probabilities = workspace .= Z ./ Z[target(ti).node]
    # return workspace .= -log.(max.(0, survival_probabilities)) ./ θ
end
function compute(::PowerMeanProximity, ti::TargetInit{<:RSP})
    θ = theta(ti)
    (; survival_probabilities, workspace) = ti
    return workspace .= survival_probabilities .^ (1 / θ)
end

_diff_CA_fun(::MinusLog) = x -> -inv(x)
_diff_AC_fun(::MinusLog) = x -> -(x)
_diff_CA_fun(::Inv) = x -> -inv(x^2) # TODO fix
_diff_AC_fun(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = -K .* x.α
_diff_KD(::Inv) = -K .^ 2

_scale_uniless!(A, ::Union{Affinity,AffinityAndCost}, g) = A .*= affinitymatrix(g)
_scale_uniless!(A, ::Union{Cost,CostAndAffinity}, g) = A .*= costmatrix(g)


# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::ConnectedHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * source_quality_spatial(init)[target] * target_quality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0
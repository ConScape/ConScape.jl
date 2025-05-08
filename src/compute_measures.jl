function get_or_compute!(ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)
    haskey(st, x) && return st[x]
    output = compute(m, ti)
    if returntrait(m) isa Union{AssignDense,SumDenseSpatial}
        st[x] = ReadOnlyArray(output)
    end
    return output
end
@inline function get_or_compute!(ti::TargetInit{<:RSP}, x::Symbol)::Vector{Float64}
    st = storage(ti)
    haskey(st, x) && return st[x]
    output = if x === :Z # "fundamental matrix"
        _fundamentalmatrix(ti)
    elseif x === :Zⁱ # elementwise inverse of Z
        ReadOnlyArray(_inv!(ti.workspace, ti.Z))
    elseif x === :Zrows
        _fundamentalrowmatrix(ti)
    elseif x === :Y # Unadjusted expected cost
        (; CW, Z, IW_factorization, workspace) = ti
        # Solve: (I - W) \ (C .* W) * Z ./ Z
        ReadOnlyArray(ldiv!(ti, IW_factorization, mul!(workspace, CW, Z)))
    elseif x === :Q
        _qualitymatrix(ti)
    elseif x === :K
        _proximitymatrix(ti)
    elseif x === :M
        _landscapematrix(ti)
    else
        error("Unknown property $x")
    end
    st[x] = output
    return output
end
@inline function get_or_compute!(ti::TargetInit{<:RandomWalk}, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]
    # Either retrieve from storage, or calculate and store
    output = if x === :Z
        _fundamentalmatrix(ti)
    elseif x === :Zⁱ
        _inv!(ti.workspace, ti.Z)
    elseif x === :Zrows
        _fundamentalrowmatrix(ti)
    elseif x === :IW_factorization
        _woodburysubtochasticmatrix(ti)
    elseif x === :IW_adj_factorization
        ti.IW_factorization'
    elseif x === :W
        # TODO less allocation
        W = copy(ti.P);
        t = target(ti).node
        W[t, :] .= 0; # set target node as killing (t row set to 0)
        W
    elseif x === :CW
        (; W) = ti
        CW = costmatrix(ti) .* W
    elseif x === :K
        _proximitymatrix(ti)
    elseif x === :M
        _landscapematrix(ti)
    elseif x === :Q
        _qualitymatrix(ti)
    elseif x === :Y # Unadjusted expected cost
        (; CW, Z, IW_factorization, workspace) = ti
        # Solve: (I - W) \ (C .* W) * Z ./ Z
        ReadOnlyArray(ldiv!(ti, IW_factorization, mul!(workspace, CW, Z)))
    else
        error("Unknown property $x")
    end
    st[x] = output
    return output
end
@inline function get_or_compute!(ti::TargetInit{<:LeastCost}, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]
    # Either retrieve from storage, or calculate and store
    output = if x == :shortest_paths
        # TODO: this is very slow, use Eikonal.jl instead
        return Graphs.dijkstra_shortest_paths(ti.cost_weighted_digraph, target(ti).node)::Graphs.DijkstraState{Float64,Int}
    elseif x == :K # "proximity vector"
        (; shortest_paths, workspace) = ti
        # TODO this should error earlier
        workspace .= distance_transformation(ti).(shortest_paths.dists)
    elseif x === :M # "landscape vector"
        _landscapematrix(ti)
    elseif x === :Q
        _qualitymatrix(ti)
    else
        error("Unknown property $x")
    end
    st[x] = output
    return output
end

# Variable generation for TargetInit
function _proximitymatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    pm = proximity_measure(ti)
    proximities = get_or_compute!(ti, pm)
    if pm isa DistanceMeasure
        dt = distance_transformation(ti)
        if !isnothing(dt)
            proximities .= dt.(proximities)
        end
    end
    proximities = _maybe_set_diagonal!(ti, proximities)
    return ReadOnlyArray(proximities)
end
function _fundamentalmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    workspace1, workspace2 = workspaces(ti)
    b = _rhs!(workspace1, nsources(ti), target(ti))
    b_copy = _rhs!(workspace2, nsources(ti), target(ti))
    return ReadOnlyArray(ldiv!(solver(ti), b, ti.IW_factorization, b_copy))
end
function _fundamentalrowmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    b = _rhs!(ti.workspace, nsources(ti), target(ti))
    return ReadOnlyArray(ldiv!(ti, ti.IW_adj_factorization, b))
end
function _landscapematrix(ti::TargetInit)
    (; qˢ, K, qᵗ, workspace) = ti
    return ReadOnlyArray(workspace .= qˢ .* K .* qᵗ)
end
function _qualitymatrix(ti::TargetInit)
    (; qˢ, qᵗ, workspace) = ti
    return ReadOnlyArray(workspace .= qˢ .* qᵗ)
end
function _woodburysubtochasticmatrix(ti::TargetInit{<:RandomWalk})
    (; P, IP, IP_factorization) = ti
    t = target(ti).node
    n = LinearAlgebra.checksquare(IP)
    # Prepare a Woodbury matrix to cheaply zero out row t, without factorization
    U = fill!(reshape(ti.workspace, (n, 1)), 0.0)
    V = fill!(reshape(ti.workspace, (1, n)), 0.0)
    U[t] = 1 # Identity
    V .= .- (IP[t:t, :])       # So that IP[t, :] + UCV[t, :] .== 0
    V[t] = -P[t, t] # So that IP[t, t] + UCV[t, t] = 1
    C = 1 # Identity
    # lu(I - W)
    # @assert IP + U * C * V .- (I - W)
    return Woodbury(IP_factorization, U, C, V)
end

# Custom `inv` broadcast that avoids Inf
_inv(Z::AbstractArray) = _inv!(similar(Z), Z)
function _inv!(Zⁱ::AbstractArray, Z::AbstractArray)
    broadcast!(Zⁱ, Z) do x
        x = inv(x)
        isfinite(x) ? x : floatmax(eltype(Z))
    end |> ReadOnlyArray
end

######################################################################################
# Proximities

function compute(::EuclidianDistance, ti::TargetInit)
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))
    return ti.workspace .= _hypot.(source_ids(ti), (target(ti).spatial,))
end

function compute(
    ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
)
    (; IW_factorization) = ti
    node = target(ti).node
    PC_rowsums = ti.workspace
    # Set target rowsum of PC to zero
    PC_rowsums .= ti.PC_rowsums
    PC_rowsums[node] = 0
    # Solve (I - W) \ PC_rowsums
    return ldiv!(ti, IW_factorization, PC_rowsums)
end

# RSP
function compute(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Y, Zⁱ) = ti
    C̄ = ti.workspace .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[target(ti).node]
    return C̄
end
function compute(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute!(ti, SurvivalProbability())
    return workspace .= -log.(max.(0, sp)) ./ θ
end
function compute(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute!(ti, SurvivalProbability())
    return workspace .= sp .^ (1 / θ)
end
function compute(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z, workspace) = ti
    return workspace .= Z ./ Z[target(ti).node]
end


######################################################################################
# Mean Kullback-Leibler Divergence

function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:LeastCost})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    output = ti.workspace
    from = Vector{Int}(undef, length(output))
    to = Vector{Int}(undef, length(output))

    n = length(from)
    dsp = dijkstra_shortest_paths(cost_weighted_digraph, target(ti).node)
    parents = dsp.parents
    # TODO explain why this is needed
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
            output[i] += -log(P[fromᵢ, toᵢ])
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
    return 0.0 # Trivially returns zero
end
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    (; θ, qˢ, qᵗ, workspace) = ti
    fed = get_or_compute!(ti, FreeEnergyDistance())
    ec = get_or_compute!(ti, ExpectedCost())
    diff = workspace .= fed .- ec
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
    node = target(ti).node
    shortest_paths_enumerated = Graphs.enumerate_paths!(path_allocs, shortest_paths, 1:length(path_allocs))
    # Set the target path to only contain itself
    targetpath = resize!(shortest_paths_enumerated[node], 1)
    targetpath[1] = node
    # Get the target weights
    weights = _weight(m, ti)
    btw = workspace .= 0.1

    @inbounds for s in eachindex(source_ids(ti))
        w = weights[s]
        for p in shortest_paths_enumerated[s]
            btw[p] += w 
        end
    end
    return btw
end
# RandomShortestPath / RandomWalk (differences are only in IW and weights)
function compute(m::Betweenness, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; Z, Zⁱ, IW_adj_factorization, workspace) = ti
    weight = _weight(m, ti)
    XZⁱt = workspace .= weight .* Zⁱ
    # Find the scaling factor: if any of XZⁱ is above 1.0 there is a risk of Inf overflow
    λ = max(1.0, maximum(XZⁱt))
    # TODO: explain what this subtraction does
    XZⁱt[target(ti).node] -= Zⁱ[target(ti).node] * sum(weight)
    # Scale MZⁱ with λ
    XZⁱtλ = XZⁱt .*= inv(λ)
    # Solve (I - W)' \ MZⁱλ, then multiply by Z and λ scaling
    return ldiv!(ti, IW_adj_factorization, XZⁱtλ) .*= λ .* Z
end

_weight(m::Union{EdgeBetweenness,Betweenness}, ti::TargetInit) = 
    _weight(weighting(m), ti)
_weight(::Unweighted, ti::TargetInit) = 1
_weight(::ProximityWeighted, ti::TargetInit) = ti.K
_weight(::QualityAndProximityWeighted, ti::TargetInit) = ti.M
_weight(::QualityWeighted, ti::TargetInit) = ti.Q
_weight(w::CustomWeighted, ti::TargetInit) = w.weight

######################################################################################
# EdgeBetweenness

# LeastCost

# TODO: implement

# RandomShortestPath / RandomWalk
function compute(::EdgeBetweenness{QualityWeighted}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; Z, Zⁱ, Zrows, W, IW_adj_factorization, qˢ, qᵗ, workspace) = ti
    QZⁱ = workspace .= qˢ .* Zⁱ .* qᵗ

    k = sum(qˢ) .* qᵗ # TODO: is this a bug? why not sum(QZⁱ) as below
    QZⁱᵀZ = ldiv!(ti, IW_adj_factorization, QZⁱ)
    RHS = QZⁱᵀZ .-= k .* Zⁱ[target(ti).node] .* Zrows
    return _combine_edge_betweenness(W, Z, RHS, target(ti))
end
function compute(
    m::EdgeBetweenness, ti::TargetInit{<:Union{RSP,RandomWalk}}
)
    (; W, Z, Zⁱ, Zrows, IW_adj_factorization, workspace) = ti
    weight = _weight(m, ti)
    MZⁱ = workspace .= weight .* Zⁱ
    k = sum(MZⁱ)
    MᵀZ = ldiv!(ti, IW_adj_factorization, MZⁱ)
    RHS = MᵀZ .-= k .* Zⁱ[target(ti).node] .* Zrows
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
function compute(m::SensitivityAnalysis{<:SourceQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; qˢ, qᵗ, K, workspace) = ti
    if sensitivitytype(m) isa Elasticity
        target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
    else # sensitivitytype(m) isa Sensitivity
        target_sensitivity = workspace .= K .* qᵗ[target(ti).node]
    end
    return target_sensitivity
end
function compute(m::SensitivityAnalysis{<:TargetQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; qˢ, qᵗ, K, workspace) = ti
    if sensitivitytype(m) isa Elasticity
        target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
    else # sensitivitytype(m) isa Sensitivity 
        target_sensitivity = workspace .= K .* qˢ
    end
    return sum(target_sensitivity)
end
function compute(m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    st = storage(ti)
    if haskey(st, :S_e_aff)
        S_e_aff, S_e_cost = st[:S_e_aff], st[:S_e_cost]
    else
        S_e_aff, S_e_cost = _permeability_sensitivity(proximity_measure(ti), ti)
        st[:S_e_aff] = S_e_aff 
        st[:S_e_cost] = S_e_cost
    end
    S_e_aff_scaled = _maybe_scale(S_e_aff, sensitivitytype(m), wrt(m), ti)
    S_e_cost_scaled = _maybe_scale(S_e_cost, sensitivitytype(m), wrt(m), ti)

    return _combine_sensitivity(wrt(m), S_e_aff_scaled, S_e_cost_scaled, ti)
end

_combine_sensitivity(::Affinity, S_e_aff, S_e_cost, ti) = S_e_aff
_combine_sensitivity(::Cost, S_e_aff, S_e_cost, ti) = S_e_cost
function _combine_sensitivity(::CostToAffinity, S_e_aff, S_e_cost, ti)
    f = _diff_CA(costfunction(ti))
    C = view(costmatrix(ti), :, target(ti).node)
    # TODO: what happens with the zeros in `costmatrix``
    # -inv(0.0) * 1.0 === NaN so we generate a lot of nans here
    return ti.workspace .= S_e_aff .+ S_e_cost .* f.(C)# .* C .!= 0
end
function _combine_sensitivity(::AffinityToCost, S_e_aff, S_e_cost, ti)
    f = _diff_AC(costfunction(ti))
    A = view(affinitymatrix(ti), :, target(ti).node)   
    return ti.workspace .= S_e_cost .+ S_e_aff .* f.(A)
end

function _permeability_sensitivity(::ExpectedCost, ti::TargetInit)
    (; θ, qˢ, qᵗ, K, Aⁱ, A_rowsums, C, Y, W, CW, IW_factorization, IW_adj_factorization, Z, Zⁱ, Zrows) = ti
    node = target(ti).node
    Md = ti.workspace .= _diff_KD(distance_transformation(ti)).(K) .* qˢ .* qᵗ
    C̄ = ti.workspace .= Y .* Zⁱ
    MdZⁱ = Md .*= Zⁱ

    MdᵀZ = ldiv!(ti, IW_factorization, MdZⁱ) # MdᵀZ = MdZⁱ' / IW
    

    k̂diagZⁱ = sum(MdZⁱ) .* Zⁱ[node]

    X3 = ti.workspace .= k̂diagZⁱ .* Zrows
    X6 = ti.workspace .= MdᵀZ .- X3

    k̂diagC̄Zⁱ = k̂diagZⁱ * C̄[node]
    RHS = vec((Md .* C̄)' .- (MdᵀZ' * CW) .+ (X3' * CW))
    X5 = ldiv!(ti, IW_adj_factorization, RHS) .- k̂diagC̄Zⁱ .* Zrows

    Wt = ti.workspace .= view(W, :, node)
    kΣ = ti.workspace .= Wt # k-weighted negative covariance matrix
    kB = ti.workspace .= Wt # k-weighted edge betweenness matrix

    for i in eachindex(Wt)
        w = Wt[i]
        # TODO explain this
        w > 0 || continue
        kB[i] *= Z[node] * X6[i]
        kΣ[i] *= Z[node] * X5[i] - Y[node] * X6[i] - C[i] * kB[i] / w
    end

    # TODO this seems wrong, dims=2 means sum does nothing
    kΣ_node = sum(kΣ; dims=2)

    S_cost = .-(kB .+ θ .* kΣ)
    S_aff = kΣ_node ./ A_rowsums .* (Wt .> 0) .- kΣ .* view(Aⁱ, :, node)

    return S_aff, S_cost
end
function _permeability_sensitivity(m::PowerMeanProximity, ti::TargetInit)
    (; θ, A, Aⁱ, A_rowsums) = ti
    node = target(ti).node

    weight = get_or_compute!(ti, m)
    bet_node_k = compute(Betweenness(CustomWeighted(weight)), ti)
    bet_edge_k = compute(EdgeBetweenness(CustomWeighted(weight)), ti)

    S_aff, S_cost = workspaces(ti)
    S_cost .= .-(bet_edge_k)
    S_aff .= (bet_edge_k .* view(Aⁱ, :, node) .-
        (bet_node_k ./ A_rowsums) .* (view(A, :, node) .> 0)) .* θ

    return S_aff, S_cost
end

_diff_CA(::MinusLog) = x -> -inv(x)
_diff_AC(::MinusLog) = x -> -(x)
_diff_CA(::Inv) = x -> -inv(x^2)
_diff_AC(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = k -> -k * x.alpha
_diff_KD(::ExpMinus) = k -> -k
_diff_KD(::Inv) = k -> -k ^ 2

# TODO is Sensitivity and Elasticity swapped here?
_maybe_scale(a, ::Sensitivity, ::Union{Affinity,AffinityToCost}, ti) =
    ti.workspace .= a .* view(affinitymatrix(ti), :, target(ti).node)
_maybe_scale(a, ::Sensitivity, ::Union{Cost,CostToAffinity}, ti) =
    ti.workspace .= a .* view(costmatrix(ti), :, target(ti).node)
_maybe_scale(a, ::Elasticity, ::Permeability, ti) = a


# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::ConnectedHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * source_quality_spatial(init)[target] * target_quality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0
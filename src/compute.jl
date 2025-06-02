function get_or_compute!(ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)
    haskey(st, x) && return st[x]
    output = compute(m, ti)
    if returntrait(m) isa ReturnDenseSpatial
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
    elseif x === :Y
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
        CW = stepcost(ti)::AbstractMatrix .* W
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
@inline function get_or_compute!(ti::TargetInit{<:LCP}, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]
    # Either retrieve from storage, or calculate and store
    output = if x == :shortest_paths
        # TODO: this is very slow, use Eikonal.jl instead
        return Graphs.dijkstra_shortest_paths(ti.cost_weighted_digraph, target(ti).node)::Graphs.DijkstraState{Float64,Int}
    elseif x == :K # "proximity vector"
        (; shortest_paths, workspace) = ti
        # TODO this should error earlier
        ReadOnlyArray(workspace .= distance_transformation(ti).(shortest_paths.dists))
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
    distances = get_or_compute!(ti, pm)
    proximities = if pm isa DistanceMeasure
        dt = distance_transformation(ti)
        if !isnothing(dt)
            ti.workspace .= dt.(distances)
        else
            distances
        end
    else
        distances
    end
    proximities = _maybe_set_diagonal!(ti, proximities)
    return ReadOnlyArray(proximities)
end
function _fundamentalmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    workspace1, workspace2 = workspaces(ti)
    n = nsources(connectedgraph(ti))
    b = _diag_vec!(workspace1, n, target(ti))
    b_copy = _diag_vec!(workspace2, n, target(ti))
    return ReadOnlyArray(ldiv!(solver(ti), b, ti.IW_factorization, b_copy))
end
function _fundamentalrowmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    b = _diag_vec!(ti.workspace, nsources(connectedgraph(ti)), target(ti))
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

#------------------------------------------------------------------------------------------
# ConnectedGraph level measures

# Compute Target level measures for a connected subgraph
function compute(m::Measure, cgi::ConnectedGraphInit)
    output = allocate_output(ConnectedGraphLevel(), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        x = compute(m, ti)
        update_output!(output, m, ti, x)
    end
    return finalize_output!(output, m, cgi)
end

# Allocate sqauare matrix of target * target size
allocate_output(l::GridGraphLevel, m::EigMax, ::ConScapeProblem, ::GridGraph, connectedgraphs::Vector) = 
    l => Vector{Tuple{Vector{Float64},Array{Float64,0},Vector{Float64}}}(undef, length(connectedgraphs))
function allocate_output(l::Level, m::EigMax, ::ConScapeProblem, ::GridGraph, cg::ConnectedGraph, precalculation)
    n = length(sourceids(cg))
    vʳ = fill(NaN, n)
    λ = fill(0.0)
    vˡ = zeros(n)
    return l => (vʳ, λ, vˡ)
end
function compute(::EigMax, ti::TargetInit)
    return nothing
    # (; M) = ti
    # K = compute(proximity_measure(ti), ti) # TODO set diag etc
    # # square submatrix defined by extracting the rows corresponding to landmarks
    # M₀₀ = view(ti.workspace, 1:length(targetids(ti)))
    # for t in targetids(ti)
    #     M₀₀[t.connectedgraphidx] = M[t.node]
    # end
    # return M₀₀
end

update_output!(output::Tuple, ::ConnectedGraphLevel, ::EigMax, ti::TargetInit, v) = nothing

# We do most of eigmax in finalize_output
function finalize_output!((vˡ, λ, vʳ), ::Level, ::EigMax, cgi::ConnectedGraphInit)
    tol = 1e-14 # TODO as a keyword somewhere
    targetnodes = map(x -> x.node, targetids(cgi))
    # square submatrix defined by extracting the rows corresponding to landmarks
    K = compute(proximity_measure(cgi), cgi) # TODO set diag etc
    M = K .*= sourcequality(cgi) .* targetquality(cgi)'
    M₀₀ = M[targetnodes, :]

    # size of the full problem
    n = nsources(connectedgraph(cgi))

    # node ids for the non-landmarks
    p₁ = setdiff(1:n, targetnodes)

    # use an Arnoldi based eigensolver to compute the largest (absolute) eigenvalue and right vector (of submatrix)
    Fps = ArnoldiMethod.partialschur(M₀₀; nev=1, tol)
    λ₀, vʳ₀ = ArnoldiMethod.partialeigen(Fps[1])

    # construct full right vector
    vʳ[targetnodes] = vʳ₀
    vʳ[p₁] = M[p₁,:] * vʳ₀ / λ₀[1]

    # compute left vector (of submatrix) by shift-invert
    Flu = lu(M₀₀ - λ₀[1] * I)
    vˡ₀ = ldiv!(Flu', rand(length(targetids(cgi))))
    rmul!(vˡ₀, inv(vˡ₀[1]))

    # construct full left vector
    vˡ[targetnodes] = vˡ₀
    λ[] = λ₀[1]

    return vˡ, λ₀[1], vʳ
end

finalize_output!(cgi) = 
    finalize_output!(outputs(cgi), measures(cgi), cgi)
function finalize_output!(outputs::NamedTuple, measures::NamedTuple, cgi)
    return map(outputs, measures) do output, measure
        finalize_output!(output, measure, cgi)
    end
end
# Trivial default finish
finalize_output!((level, output)::Pair, m::Measure, sgi::ConnectedGraphInit) = finalize_output!(output, level, m, sgi)
finalize_output!(output, level::Level, ::Measure, ::ConnectedGraphInit) = output

# EdgeBetweenness

allocate_output(l::GridGraphLevel, ::EdgeBetweenness, ::ConScapeProblem, ::GridGraph, connectedgraphs::Vector) = 
    l => Vector{SparseMatrixCSC{Float64,Int}}(undef, length(connectedgraphs))
allocate_output(l::ConnectedGraphLevel, ::EdgeBetweenness, ::ConScapeProblem, ::GridGraph, ::ConnectedGraph, precalculation) = 
    l => mapnz(_ -> 0.0, precalculation.W)

# RandomShortestPath / RandomWalk
function compute(
    m::EdgeBetweenness, ti::TargetInit{<:Union{RSP,RandomWalk}}
)
    (; Z, Zⁱ, Zrows, IW_adj_factorization, workspace) = ti
    node = target(ti).node
    weight = _weight(m, ti)
    XZⁱ = workspace .= weight .* Zⁱ
    x = sum(weight)
    XᵀZ = ldiv!(ti, IW_adj_factorization, XZⁱ)
    XᵀZ .-= x .* Zⁱ[node] .* Zrows
    return (; Z, XᵀZ)
end

function update_output!(output::SparseMatrixCSC, ::ConnectedGraphLevel, ::EdgeBetweenness, ti::TargetInit, v::NamedTuple)
    (; Z, XᵀZ) = v
    (; W) = ti
    foreachnz(W) do i, j, n
        @inbounds output.nzval[n] += W.nzval[n] * Z[j] * XᵀZ[i]
    end
    return output
end

function transfer_output!(dest::SparseMatrixCSC, source::SparseMatrixCSC, m::EdgeBetweenness, cgi::ConnectedGraphInit)
    dest[LinearIndices(size(cgi))[sourceids(cgi)], map(t -> t.gridgraphidx, targetids(cgi))] .= source
end



# LeastCostPath EdgeBetweenness
# Not implemented


######################################################################################
# Proximities

function compute(::Distance, ti::TargetInit{<:Euclidean})
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))
    return ti.workspace .= _hypot.(sourceids(ti), (target(ti).spatialidx,))
end
compute(::Distance, ti::TargetInit{<:LCP}) =
    ReadOnlyArray(ti.shortest_paths.dists)

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

function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:LCP})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    output = ti.workspace
    from = Vector{Int}(undef, length(output))
    to = Vector{Int}(undef, length(output))

    n = length(from)
    dsp = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, target(ti).node)
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
# FunctionalHabitat 

compute(::FunctionalHabitat, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

# This differs form FunctionalHabitat in that it returns the full size matrix
compute(::LandscapeMatrix, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

######################################################################################
# Betweenness

# LeastCostPath
function compute(m::Betweenness, ti::TargetInit{<:LCP})
    (; shortest_paths, path_allocs, workspace) = ti
    node = target(ti).node
    shortest_paths_enumerated = Graphs.enumerate_paths!(path_allocs, shortest_paths, 1:length(path_allocs))
    # Set the target path to only contain itself
    targetpath = resize!(shortest_paths_enumerated[node], 1)
    targetpath[1] = node
    # Get the target weights
    weights = _weight(m, ti)
    btw = workspace .= 0.1

    @inbounds for s in eachindex(sourceids(ti))
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
    isnothing(weight) && error()
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
    return target_sensitivity
end
compute(m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:Union{RSP,RandomWalk}}) = 
    compute(proximity_measure(ti), m, ti)
function compute(::ExpectedCost, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit)
    (; K, qˢ, qᵗ, Y, CW, IW_adj_factorization, Z, Zⁱ, Zrows) = ti
    node = target(ti).node
    Kd = ti.workspace .= _diff_KD(distance_transformation(ti)).(K)
    Md = ti.workspace .= Kd .* qˢ .* qᵗ
    m = sum(Md)
    MdZⁱ = Md .* Zⁱ
    MdᵀZ = ti.workspace .= MdZⁱ 
    ldiv!(ti, IW_adj_factorization, MdᵀZ) # MdᵀZ = MdZⁱ' / IW
    k̂diagZⁱ = m * Zⁱ[target(ti).node]

    X3 = ti.workspace .= k̂diagZⁱ .* Zrows
    X6 = ti.workspace .= MdᵀZ .- X3
    C̄ᵣ = ti.workspace .= Y .* Zⁱ

    k̂diagC̄Zⁱ = k̂diagZⁱ * C̄ᵣ[node]
    RHS = vec(MdZⁱ .* C̄ᵣ) .- vec((MdᵀZ' * CW) .+ (X3' * CW))
    X5 = ldiv!(ti, IW_adj_factorization, RHS) .- k̂diagC̄Zⁱ .* Zrows

    return (; Z, Y, X5, X6)
end
function compute(pmp::PowerMeanProximity, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit)
    (; Z, θ) = ti
    weight = ti.workspace .= Z ./= Z[target(ti).node] .^ θ
    bet_node_k = compute(Betweenness(CustomWeighted(weight)), ti)
    bet_edge_k = compute(EdgeBetweenness(CustomWeighted(weight)), ti)
    return (; bet_node_k, bet_edge_k)
end


# kB and kΣ are set up as zeroed-out W matrices
allocate_output(l::GridGraphLevel, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, args...) = 
    allocate_output(l, ReturnDenseSpatialSum(), p, args...)
allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, args...) = 
    allocate_output(l, proximity_measure(p), m, p, args...)
function allocate_output(l::Level, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, gg::GridGraph, cg::ConnectedGraph, precalculation)
    (; W) = precalculation
    bet_edge_k = allocate_output(l, EdgeBetweenness(CustomWeighted(nothing)), p, gg, cg, precalculation)
    bet_node_k = allocate_output(l, Betweenness(CustomWeighted(nothing)), p, gg, cg, precalculation)
    result = mapnz(_ -> 0.0, W)
    resultrows = zeros(size(W, 1))
    return l => (; bet_node_k, bet_edge_k, result, resultrows)
end
# kB and kΣ are set up as zeroed-out W matrices
function allocate_output(l::Level, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, ::ConScapeProblem, ::GridGraph, ::ConnectedGraph, precalculation)
    (; W) = precalculation
    kB = mapnz(_ -> 0.0, W)
    kΣ = mapnz(_ -> 0.0, W)
    result = mapnz(_ -> 0.0, W)
    resultrows = zeros(size(W, 1))
    return l => (; kB, kΣ, result, resultrows)
end

# And we store into them for each target
update_output!(output::NamedTuple, l::ConnectedGraphLevel, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v) =
    update_output!(output, l, proximity_measure(ti), m, ti, v)
function update_output!(output, l::ConnectedGraphLevel, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v)
    (; W, C) = ti
    (; kB, kΣ) = output
    (; Z, Y, X5, X6) = v

    foreachnz(W) do i, j, n
        kB.nzval[n] += W.nzval[n] * Z[j] * X6[i]
        kΣ.nzval[n] += Z[j] * X5[i] - Y[j] * X6[i] - C.nzval[n] * kB.nzval[n] / W.nzval[n]
    end
    return output
end
function update_output!(output, l::ConnectedGraphLevel, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v)
    (; bet_edge_k, bet_node_k) = v
    # Just call update_output! on the component parts
    update_output!(output.bet_edge_k, EdgeBetweenness(CustomWeighted(nothing)), ti, bet_edge_k)
    update_output!(output.bet_node_k, Betweenness(CustomWeighted(nothing)), ti, bet_node_k)
    return nothing
end

# And after all targets run, finalize them 
finalize_output!(output::Pair, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit) =
    finalize_output!(output[2], proximity_measure(sgi), m, sgi)
function finalize_output!(output::NamedTuple, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit)
    (; kB, kΣ, result) = output
    (; A_rowsums, Aⁱ) = precalculation(sgi)

    kΣ_node = sum(kΣ, dims=2)
    foreachnz(Aⁱ) do i, j, n 
        S_cost = kB.nzval[n] + theta(sgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[j] / A_rowsums[j]) * 1 - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), sgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), sgi, n)
        result.nzval[n] = _combine_sensitivity(wrt(m), S_e_likelihood, S_e_cost_scaled, sgi, n)
    end

    return output
end
function finalize_output!(output::NamedTuple, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit)
    (; bet_edge_k, bet_node_k, result, resultrows) = output
    (; A_rowsums, Aⁱ) = precalculation(sgi)
    A = steplikelihood(sgi)
    foreachnz(Aⁱ) do i, j, n 
        S_cost = bet_edge_k[2].nzval[n]
        S_likelihood = bet_edge_k[2].nzval[n] * Aⁱ.nzval[n] - bet_node_k[2][j] / A_rowsums[j] * A.nzval[n] * theta(sgi)
        S_e_likelihood = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), sgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), sgi, n)
        result.nzval[n] = _combine_sensitivity(wrt(m), S_e_likelihood, S_e_cost_scaled, sgi, n)
    end
    # TODO dont allocate
    return resultrows .= vec(sum(result; dims=1))
end

transfer_output!(dest::AbstractMatrix, source::NamedTuple, ::SensitivityAnalysis, cgi::ConnectedGraphInit) =
    dest[sourceids(cgi)] .= source.resultrows

_combine_sensitivity(::StepLikelihood, S_e_likelihood, S_e_cost, ti, n) = S_e_likelihood
_combine_sensitivity(::StepCost, S_e_likelihood, S_e_cost, ti, n) = S_e_cost
function _combine_sensitivity(::StepCostToLikelihood, S_e_likelihood, S_e_cost, ti, n)
    f = _diff_CA(costfunction(ti))
    C = stepcost(ti)
    S_e_likelihood + S_e_cost * f(C.nzval[n])
end
function _combine_sensitivity(::StepLikelihoodToCost, S_e_likelihood, S_e_cost, ti, n)
    f = _diff_AC(costfunction(ti))
    L = steplikelihood(ti)
    S_e_cost + S_e_likelihood * f(L.nzval[n])
end

_maybe_scale(a, ::Elasticity, ::Union{StepLikelihood,StepLikelihoodToCost}, ti, n) =
    a * steplikelihood(ti).nzval[n]
_maybe_scale(a, ::Elasticity, ::Union{StepCost,StepCostToLikelihood}, ti, n) =
    a * stepcost(ti).nzval[n]
_maybe_scale(a, ::Sensitivity, ::Permeability, ti, n) = a

_diff_CA(::MinusLog) = x -> -inv(x)
_diff_AC(::MinusLog) = x -> -(x)
_diff_CA(::Inv) = x -> -inv(x^2)
_diff_AC(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = k -> -k * x.alpha
_diff_KD(::ExpMinus) = k -> -k
_diff_KD(::Inv) = k -> -k ^ 2

# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::FunctionalHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * sourcequality_spatial(init)[target] * targetquality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0

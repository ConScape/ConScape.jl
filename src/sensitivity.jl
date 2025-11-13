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

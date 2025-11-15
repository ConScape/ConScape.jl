const store = Ref{Any}()

######################################################################################
# Sensitivity
# TODO: finish and test this fragment
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

# Shared sequence
allocate_output(l::GridGraphLevel, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, args...) =
    allocate_output(l, ReturnDenseSpatialSum(), p, args...)
# Allocate output and intermediate storage for the specific proximity_measure
allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, args...) =
    allocate_output(l, proximity_measure(p), m, p, args...)
# Compute for the specific proximity_measure
compute(m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:Union{RSP,RandomWalk}}) =
    compute(proximity_measure(ti), m, ti)
# Store the output of compute into the output for each target
update_output!(output::NamedTuple, l::ConnectedGraphLevel, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v) =
    update_output!(output, l, proximity_measure(ti), m, ti, v)
# Finalize the output for the proximity_measure after all targets run
finalize_output!(output::Pair, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit) =
    finalize_output!(output[2], proximity_measure(sgi), m, sgi)

# Allocate for ExpectedCost
function allocate_output(l::Level, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, ::ConScapeProblem, ::GridGraph, cg::ConnectedGraph, precalculation)
end
function allocate_storage(m::SensitivityAnalysis{<:Permeability}, ::GridGraph, cg::ConnectedGraphInit)
    (; W) = precalculation

    # kB and kΣ are set up as zeroed-out W matrices
    kB = mapnz(_ -> 0.0, W)
    kΣ = mapnz(_ -> 0.0, W)
    result = mapnz(_ -> 0.0, W)
    resultrows = zeros(size(W, 1))
    mdiagZⁱ = buidl_mdiagZⁱ(cg)

    store[] = (; 
        W,
        M=zeros(size(W)),
        K=zeros(size(W)),
        Z=zeros(size(W)),
        Zⁱ=zeros(size(W)),
        Zrows=zeros(size(W)),
        X3=zeros(size(W)),
        X6=zeros(size(W)),
        mdiagZⁱ=zeros(size(W, 1)),
    )

    return (; kB, kΣ, result, resultrows, mdiagZⁱ)
end
# For ExpectedCost sensitivity we need to loop over Z calculations twice
# Once here to build the diagonal vector that we need later
function buidl_mdiagZⁱ(cg)
    mdiagZⁱ = zeros(length(targetids(cg)))
    for target_id in targetids(cg)
        # Precalculate for this target and graph measures
        ti = init(cg, target_id)
        dt = distance_transformation(ti)
        (; K, qˢ, qᵗ, Zⁱ) = ti
        Kd = ti.workspace .= _diff_KD(dt).(K)
        Md = ti.workspace .= qˢ .* Kd .* qᵗ
        m = sum(Md)
        # Surely there is a faster way to do this...
        mdiagZⁱ[i] = m * Zⁱ[target(ti).node]
    end
    return mdiagZⁱ 
end
# Compute for ExpectedCost
# We have to do all compute in update_output! so we have the diagonal available
compute(::ExpectedCost, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit) = nothing
# Store the output of compute into the output for each target for ExpectedCost
function update_output!(output, l::ConnectedGraphLevel, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v)
    (; W, C, CW, Z, K, qˢ, qᵗ, Y, IW_adj_factorization, Zrows, Z, Zⁱ) = ti
    (; mdiagZⁱ, kB, kΣ) = output
    node = target(ti).node

    Kd = ti.workspace .= _diff_KD(distance_transformation(ti)).(K)
    Md = ti.workspace .= qˢ .* Kd .* qᵗ
    MdZⁱ = ti.workspace .= Md .* Zⁱ
    MdᵀZ = ti.workspace .= MdZⁱ
    C̄ᵣ = ti.workspace .= Y .* Zⁱ

    m = sum(Md)

    ldiv!(ti, IW_adj_factorization, MdᵀZ) # MdᵀZ = MdZⁱ' / IW

    # store[].Zrows[:, node] .= Zrows
    # store[].X3[:, node] .= X3
    # store[].X6[:, node] .= X6
    # store[].mdiagZⁱ[node] = mdiagZⁱ
    # store[].K[:, node] .= Kd 
    # store[].M[:, node] .= Md 
    # store[].Z[:, node] .= Z
    # store[].Zⁱ[:, node] .= inv.(Z)

    foreachnz(W) do i, j, n
        kBn = W.nzval[n] * (Z[j] * X6[i])
        kB.nzval[n] += kBn 
        kΣ.nzval[n] += W.nzval[n] * ((Z[j] * X5[i]) - (Y[j] * X6[i]) - C.nzval[n] * kBn / W.nzval[n])
        # kΣ.nzval[n] += W.nzval[n] * ((Z[j]))# - (Y[j] * X6[i]) - C.nzval[n] * kBn / W.nzval[n])
    end

    return output
end
# Finalize for ExpectedCost after all targets run
# TODO: use cgi instead of sgi
function finalize_output!(output::NamedTuple, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit)
    (; kB, kΣ, result, resultrows) = output
    (; A_rowsums, Aⁱ) = precalculation(sgi)

    kΣ_node = sum(kΣ, dims=2)
    foreachnz(Aⁱ) do i, j, n
        S_cost = kB.nzval[n] + theta(sgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[j] / A_rowsums[j]) * 1 - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), sgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), sgi, n)
        result.nzval[n] = _combine_sensitivity(wrt(m), S_e_likelihood, S_e_cost_scaled, sgi, n)
    end
    # TODO what to do here
    resultrows .= vec(sum(result; dims=1))
    
    println("storing S_e_aff and S_e_cost...")
    store[] = merge(store[], (; kB, kΣ))

    return output
end
# Transfer connected graph output to the final spatial grid
transfer_output!(dest::AbstractMatrix, source::NamedTuple, ::SensitivityAnalysis, cgi::ConnectedGraphInit) =
    dest[sourceids(cgi)] .= source.resultrows


# Allocate output for PowerMeanProximity
function allocate_output(l::Level, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, gg::GridGraph, cg::ConnectedGraph, precalculation)
    (; W) = precalculation

    custom_weighted = CustomWeighted(nothing) # We dont need the weights in the allocation phase, just the type
    bet_edge_k_output = allocate_output(l, EdgeBetweenness(custom_weighted), p, gg, cg, precalculation)
    bet_node_k_output = allocate_output(l, Betweenness(custom_weighted), p, gg, cg, precalculation)
    # Result is the same size sparse array as W
    result = mapnz(_ -> 0.0, W)
    # We will sum result into these rows
    resultrows = zeros(size(W, 1))

    return l => (; bet_node_k_output, bet_edge_k_output, result, resultrows)
end
# Store the output of compute into the output for each target for PowerMeanProximity
# This is part of PM_sensitivity in the original code
function compute(pmp::PowerMeanProximity, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit)
    (; qˢ, Z, qᵗ, θ, workspace) = ti
    # Calculate weights for this target
    weights = workspace .= (qˢ .* ((Z ./ Z[target(ti).node]) .^ θ) .* qᵗ)
    custom_weighted = CustomWeighted(weights)

    # Compute node and edge betweenness for these weights
    bet_node_k_computed = compute(Betweenness(custom_weighted), ti)
    bet_edge_k_computed = compute(EdgeBetweenness(custom_weighted), ti)

    return (; bet_node_k_computed, bet_edge_k_computed)
end
function update_output!(output, l::ConnectedGraphLevel, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit, v)
    (; bet_edge_k_computed, bet_node_k_computed) = v
    # Just call update_output! on the component parts
    custom_weighted = CustomWeighted(nothing) # We dont need the weights for the update
    update_output!(output.bet_edge_k_output, EdgeBetweenness(custom_weighted), ti, bet_edge_k_computed)
    update_output!(output.bet_node_k_output, Betweenness(custom_weighted), ti, bet_node_k_computed)

    return nothing
end
# Finalize for PowerMeanProximity after all targets run
function finalize_output!(output::NamedTuple, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit)
    (; bet_edge_k_output, bet_node_k_output, result, resultrows) = output
    (; A_rowsums, Aⁱ) = precalculation(sgi)
    A = steplikelihood(sgi)
    bet_edge_k, bet_node_k = bet_edge_k_output[2], bet_node_k_output[2]
    
    Idx = A.>0
    Aⁱ = ConScape.mapnz(inv, A)
    S_e_aff = (bet_edge_k .* Aⁱ .- (bet_node_k[sourceids(sgi)] ./ A_rowsums) .* Idx) .* theta(sgi)
    S_e_cost = -bet_edge_k
    # This is from PM_sensitivy in the original code
    foreachnz(Aⁱ) do i, j, n
        I = sourceids(sgi)[i]
        S_cost = -bet_edge_k.nzval[n]
        S_likelihood = (bet_edge_k.nzval[n] * Aⁱ.nzval[n] - (bet_node_k[I] / A_rowsums[i]) * (A.nzval[n] > 0)) * theta(sgi)
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), sgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), sgi, n)
        x = _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, sgi, n)
        result.nzval[n] = x
    end

    # PERFORMANCE: dont allocate in sum
    resultrows .= vec(sum(result; dims=1))

    return output
end

_combine_sensitivity(::StepLikelihood, S_e_likelihood, S_e_cost, ti, n) = S_e_likelihood
_combine_sensitivity(::StepCost, S_e_likelihood, S_e_cost, ti, n) = S_e_cost
function _combine_sensitivity(::StepCostToLikelihood, S_e_likelihood, S_e_cost, ti, n)
    f = _diff_CA(costfunction(ti))
    L = steplikelihood(ti)

    return S_e_likelihood + S_e_cost * f(L.nzval[n])
end
function _combine_sensitivity(::StepLikelihoodToCost, S_e_likelihood, S_e_cost, ti, n)
    f = _diff_AC(costfunction(ti))
    L = steplikelihood(ti)

    return S_e_cost + S_e_likelihood * f(L.nzval[n])
end

# Apply scale for Elasticity, or not for Sensitivity. 
# Corresponds to unitless=true/false in the old code.
_maybe_scale(a, ::Elasticity, ::Union{StepLikelihood,StepLikelihoodToCost}, ti, n) =
    a * steplikelihood(ti).nzval[n]
_maybe_scale(a, ::Elasticity, ::Union{StepCost,StepCostToLikelihood}, ti, n) =
    a * stepcost(ti).nzval[n]
_maybe_scale(a, ::Sensitivity, ::Permeability, ti, n) = a

# TODO: CL/LC for likelihood not affinity
_diff_CA(::MinusLog) = x -> -inv(x)
_diff_AC(::MinusLog) = x -> -(x)
_diff_CA(::Inv) = x -> -inv(x^2)
_diff_AC(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = k -> -k * x.alpha
_diff_KD(::ExpMinus) = k -> -k
_diff_KD(::Inv) = k -> -k ^ 2

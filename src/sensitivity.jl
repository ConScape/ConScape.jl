const store = Ref{Any}()

######################################################################################
# Sensitivity
# TODO: finish and test this fragment
function compute!(output, m::SensitivityAnalysis{<:SourceQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; qˢ, qᵗ, K, workspace) = ti
    if sensitivitytype(m) isa Elasticity
        target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
    else # sensitivitytype(m) isa Sensitivity
        target_sensitivity = workspace .= K .* qᵗ[target(ti).node]
    end
    return target_sensitivity
end
function compute!(output, m::SensitivityAnalysis{<:TargetQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; qˢ, qᵗ, K, workspace) = ti
    if sensitivitytype(m) isa Elasticity
        target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
    else # sensitivitytype(m) isa Sensitivity
        target_sensitivity = workspace .= K .* qˢ
    end
    return target_sensitivity
end

# Shared sequence

# GridGraph behavior
allocate_output(l::Level, ::ReturnCustom, m::SensitivityAnalysis, p::ConScapeProblem, args...) =
    allocate_output(l, ReturnDenseSpatialSum(), m, p, args...)
# ConnectedGraph behavior
function allocate_output(
    l::Level, ::ReturnCustom, ::SensitivityAnalysis, ::ConScapeProblem, ::GridGraph, ::ConnectedGraph, precalculation::NamedTuple
)
    l => zeros(size(precalculation.W, 1))
end

allocate_intermediate(m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit) =
    allocate_intermediate(proximity_measure(cgi), m, cgi)
# Compute for the specific proximity_measure
compute!(output::Pair, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:Union{RSP,RandomWalk}}) =
    compute_sensitivity!(output, proximity_measure(ti), m, ti)
# Finalize the output for the proximity_measure after all targets run
    #
function finalize_output!((level, output)::Pair, m::SensitivityAnalysis, sgi::ConnectedGraphInit, intermediates)
    println("finalizing sensitivity...")
    finalize_output!(output, proximity_measure(sgi), m, sgi, intermediates)
end

# Allocate storage for ExpectedCost
function allocate_intermediate(::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit)
    (; W, C, CW) = precalculation(cgi)

    # kB and kΣ are set up as zeroed-out W matrices
    intermediates = (; 
        # Keep the sparse structure but initialise values to zero internally
        kB = mapnz(_ -> 0.0, W),
        kΣ = mapnz(_ -> 0.0, W),
        result = mapnz(_ -> 0.0, W),
        mdiagZⁱ = zeros(length(targetids(cgi))),
        mdiagC̄Zⁱ = zeros(length(targetids(cgi))),
        Zm = zeros(size(W)), 
        Zrowsm = zeros(size(W)),
        Ym = zeros(size(W)),
        RHS=zeros(size(W)),
        RHScopy=zeros(size(W)),
    )
    
    for target in targetids(cgi)
        # Precalculate for this target and graph measures
        ti = TargetInit(cgi, target)
        (; K, qˢ, qᵗ, Y, Z, Zⁱ) = ti
        dt = distance_transformation(ti)
        node = target.node
        idx = target.connectedgraphidx

        Kd = ti.workspace .= _diff_KD(dt).(K)
        Md = ti.workspace .= qˢ .* Kd .* qᵗ
        mdiagZⁱ = sum(Md) * Zⁱ[node]
        intermediates.Zm[:, node] .= Z
        intermediates.Ym[:, node] .= Y
        intermediates.mdiagZⁱ[node] = mdiagZⁱ
        intermediates.mdiagC̄Zⁱ[node] = mdiagZⁱ * Zⁱ[idx] * Y[idx]
    end

    store[] = (; 
        W,
        C,
        CW,
        M=zeros(size(W)),
        K=zeros(size(W)),
        Z=zeros(size(W)),
        Zⁱ=zeros(size(W)),
        Zrows=zeros(size(W)),
        X3=zeros(size(W)),
        X5=zeros(size(W)),
        X6=zeros(size(W)),
        MdᵀZ=zeros(size(W)),
        MdZⁱC̄ᵣⁱ=zeros(size(W)),
        MdᵀZCW=zeros(size(W)),
        X3CW=zeros(size(W)),
        intermediates...
    )

    return intermediates
end
# For ExpectedCost sensitivity we need to loop over Z calculations twice
# Once here to build the diagonal vector that we need later
# Compute for ExpectedCost
# We have to do all compute in update_output! so we have the diagonal available
function compute_sensitivity!(
    output::Pair, ::ExpectedCost, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit
)
    (; W, C, CW, K, qˢ, qᵗ, IW_adj_factorization, mdiagZⁱ, Zm, Ym, Zrows, Zrowsm, kB, kΣ, RHS) = ti
    node = target(ti).node

    # Get stored Z and Y rather than calculating again
    Z = workspace(ti) .= view(Zm, :, node)
    Y = workspace(ti) .= view(Ym, :, node)

    Zrowsm[:, node] .= Zrows
    X3 = workspace(ti) .= mdiagZⁱ .* Zrows
    Zⁱ = workspace(ti) .= _inv.(Z)

    Kd = workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
    Md = workspace(ti) .= qˢ .* Kd .* qᵗ
    MdZⁱ = workspace(ti) .= Md .* Zⁱ
    MdᵀZ = ldiv!(ti, IW_adj_factorization, (workspace(ti) .= MdZⁱ)) # MdᵀZ = MdZⁱ' / IW
    # MZⁱ = Md .* Zⁱ # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ

    C̄ᵣ = workspace(ti) .= Y .* Zⁱ
    view(RHS, :, node) .+= Md .* Zⁱ .* C̄ᵣ
    foreachnz(CW) do i, j, n
        if i == node
            # The contribution is the target column of X3, scaled by the value.
            for k in 1:length(X3)
                RHS[j, k] += X3[k] * CW.nzval[n]
            end
        end
        RHS[j, node] -= MdᵀZ[i] .* CW.nzval[n]
    end
        # The contribution is the i-th column of M, scaled by the value.
    # RHS[:, node] -= MdᵀZ .* CW[:, node]
    # MᵀZCW = (MᵀZ * CW) 
    # X3CW = (X3 * CW)
    # RHS = MZⁱ.* C̄ᵣ .- MᵀZCW' .+ X3CW'

    # foreachnz(W) do i, j, n
        # Manual matmul for this column
        # kBn = W.nzval[n] * Z[j] * X6[i]
        # kB.nzval[n] += kBn 
        # kΣ.nzval[n] += W.nzval[n] * ((Z[j] * X5[i]) - (Y[j] * X6[i]) - C.nzval[n] * kBn / W.nzval[n])
        # kΣ.nzval[n] += W.nzval[n] * ((Z[j]))# - (Y[j] * X6[i]) - C.nzval[n] * kBn / W.nzval[n])
    # end
    
    # (; W, C, CW, K, qˢ, qᵗ, Y, IW_adj_factorization, Zrows, Z, Zⁱ, mdiagZⁱ, mdiagC̄Zⁱ, kB, kΣ) = ti
    # node = target(ti).node
    #
    # Kd = workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
    # Md = workspace(ti) .= qˢ .* Kd .* qᵗ
    # MdZⁱ = workspace(ti) .= Md .* Zⁱ
    # C̄ᵣ = workspace(ti) .= Y .* Zⁱ
    # MdᵀZ = ldiv!(ti, IW_adj_factorization, (workspace(ti) .= MdZⁱ)) # MdᵀZ = MdZⁱ' / IW
    #
    # X3 = workspace(ti) .= mdiagZⁱ .* Zrows
    # @show size(MdᵀZ) size(CW)
    MdᵀZCW = workspace(ti) .= vec(MdᵀZ' * CW')
    X3CW = workspace(ti) .= vec(CW' * X3)
    MdZⁱC̄ᵣ = (workspace(ti) .= MdZⁱ .* C̄ᵣ) 
    # RHS = workspace(ti) .= MdZⁱC̄ᵣ .- MdᵀZCW .+ X3CW
    # X5 = ldiv!(ti, IW_adj_factorization, RHS) .-= (mdiagC̄Zⁱ .* Zrows)
    #
    # X6 = workspace(ti) .= MdᵀZ .- X3
    #
    # store[].Zrows[:, node] .= Zrows
    store[].X3[:, node] .= X3
    # store[].X5[:, node] .= X5
    store[].MdᵀZ[:, node] .= MdᵀZ
    # store[].K[:, node] .= Kd 
    # store[].M[:, node] .= Md 
    # store[].Z[:, node] .= Z
    # store[].Zⁱ[:, node] .= Zⁱ
    store[].MdZⁱC̄ᵣⁱ[:, node] .= MdZⁱC̄ᵣ 
    store[].MdᵀZCW[:, node] .= MdᵀZCW
    store[].X3CW[:, node] .= X3CW
    # store[].RHS[:, node] .= RHS
    #
    # update_ks!(kB, kΣ, W, Z, C, Y, X5, X6, node)
end

# Finalize for ExpectedCost after all targets run
# TODO: use cgi instead of sgi
function finalize_output!(
    output::AbstractArray, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit, intermediates::NamedTuple
)
    (; W, C, CW, IW_adj_factorization, A_rowsums, Aⁱ) = cgi
    (; kB, kΣ, mdiagZⁱ, mdiagC̄Zⁱ, result, Ym, Zm, Zrowsm, RHS, RHScopy) = intermediates
    RHScopy .= RHS

    free!(workspaces(cgi))

    MᵀZ = store[].MdᵀZ
    X5m = ldiv!(solver(cgi), RHS, IW_adj_factorization, RHScopy)

    cur_j = 1
    Z = workspace(cgi) .= view(Zm, 1, :)
    Y = workspace(cgi) .= view(Ym, 1, :)
    X5 = workspace(cgi)
    X6 = workspace(cgi)
    foreachnz(W) do i, j, n
        if j != cur_j
            Z .= view(Zm, j, :)
            Y .= view(Ym, j, :)
            cur_j = j
        end
        @views X5 .= X5m[i, :] .- mdiagC̄Zⁱ .* Zrowsm[:, i] # "X1- X2 - X4"
        @views X6 .= MᵀZ[i, :] .- mdiagZⁱ .* Zrowsm[:, i]
        kB.nzval[n] = W.nzval[n] * ((Z' * X6)[1])
        kΣ.nzval[n] = W.nzval[n] * ((Z' * X5)[1] - (Y' * X6)[1] - C.nzval[n] * kB.nzval[n] / W.nzval[n])
    end

    kΣ_node = sum(kΣ, dims=2)
    foreachnz(Aⁱ) do i, j, n
        S_cost = kB.nzval[n] + theta(cgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[j] / A_rowsums[j]) * 1 - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        result.nzval[n] = _combine_sensitivity(wrt(m), S_e_likelihood, S_e_cost_scaled, cgi, n)
    end

    store[] = merge(store[], 
        (; W, C, CW, Z=Zm, Zrows=Zrowsm, Y=Ym,
            kB, kΣ, mdiagZⁱ, mdiagC̄Zⁱ, MᵀZ, RHS
        )
    )
    # TODO what to do here
    output .= vec(sum(result; dims=1))

    return output
end
# Transfer connected graph output to the final spatial grid
transfer_output!(dest::AbstractMatrix, source, ::SensitivityAnalysis, cgi::ConnectedGraphInit) =
    dest[sourceids(cgi)] .= source

function allocate_intermediate(::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit)
    (; W, C) = cgi

    custom_weighted = CustomWeighted(nothing) # We dont need the weights in the allocation phase, just the type
    bet_edge_k = allocate_intermediate(EdgeBetweenness(custom_weighted), cgi) 
    bet_edge_k_output = allocate_output(ConnectedGraphLevel(), EdgeBetweenness(custom_weighted), cgi) 
    bet_node_k_output = allocate_output(ConnectedGraphLevel(), Betweenness(custom_weighted), cgi)
    result = mapnz(_ -> 0.0, W)
    intermediates = (; bet_edge_k, bet_edge_k_output, bet_node_k_output, result)

    return intermediates
end
const WEIGHT = Ref{Any}()
# Store the output of compute into the output for each target for PowerMeanProximity
# This is part of PM_sensitivity in the original code
function compute_sensitivity!(output, pmp::PowerMeanProximity, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit)
    (; qˢ, Z, qᵗ, θ) = ti
    node = target(ti).node
    # Calculate weights for this target
    weights = readonlyarray(workspace(ti) .= (qˢ .* ((Z ./ Z[node]) .^ θ) .* qᵗ))
    store[].K[:, node] .= ((Z ./ Z[node]) .^ θ)
    store[].M[:, node] .= weights

    # Compute node and edge betweenness for these weights
    bet_node = Betweenness(CustomWeighted(weights))
    bet_edge = EdgeBetweenness(CustomWeighted(weights))
    ti_bet_node = rebuild(ti; intermediates=(;))
    ti_bet_edge = rebuild(ti; intermediates=intermediates(ti).bet_edge_k)
    compute!(ti.bet_edge_k_output, bet_edge, ti_bet_edge)
    compute!(ti.bet_node_k_output, bet_node, ti_bet_node)

    return output
end
 
# Finalize for PowerMeanProximity after all targets run
function finalize_output!(
    output::AbstractArray, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, sgi::ConnectedGraphInit, intermediates
)
    (; A_rowsums, Aⁱ, A) = sgi
    (; bet_edge_k_output, bet_node_k_output, result) = intermediates
    bet_edge_k, bet_node_k = bet_edge_k_output[2], bet_node_k_output[2]

    # Finalize edge betweenness
    bet_edge = EdgeBetweenness(CustomWeighted(nothing))
    finalize_output!(intermediates.bet_edge_k_output, bet_edge, sgi, intermediates.bet_edge_k)
    
    # Idx = A.>0
    # Aⁱ = ConScape.mapnz(inv, A)
    # S_e_aff = (bet_edge_k .* Aⁱ .- (bet_node_k[sourceids(sgi)] ./ A_rowsums) .* Idx) .* theta(sgi)
    # S_e_cost = -bet_edge_k

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

    # PERFormance: dont allocate in sum
    output .= vec(sum(result; dims=1))
    # store[] = merge(store[], (; bet_edge_k, bet_node_k, S_e_aff, S_e_cost))

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

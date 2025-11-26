const store = Ref{Any}((;))

######################################################################################
# Sensitivity
# TODO: finish and test this fragment
# function compute!(output, m::SensitivityAnalysis{<:SourceQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
#     (; qˢ, qᵗ, K, workspace) = ti
#     if sensitivitytype(m) isa Elasticity
#         target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
#     else # sensitivitytype(m) isa Sensitivity
#         target_sensitivity = workspace .= K .* qᵗ[target(ti).node]
#     end
#     return target_sensitivity
# end
# function compute!(output, m::SensitivityAnalysis{<:TargetQuality}, ti::TargetInit{<:Union{RSP,RandomWalk}})
#     (; qˢ, qᵗ, K, workspace) = ti
#     if sensitivitytype(m) isa Elasticity
#         target_sensitivity = workspace .*= qˢ .* K .* qᵗ[target(ti).node]
#     else # sensitivitytype(m) isa Sensitivity
#         target_sensitivity = workspace .= K .* qˢ
#     end
#     return target_sensitivity
# end

# Shared sequence

# GridGraph behavior
allocate_output(l::Level, ::ReturnCustom, m::SensitivityAnalysis, p::ConScapeProblem, args...) =
    allocate_output(l, ReturnDenseSpatialSum(), m, p, args...)
# ConnectedGraph behavior
function allocate_output(
    l::Level, ::ReturnCustom, ::SensitivityAnalysis, ::ConScapeProblem, ::GridGraph, cg::ConnectedGraph, precalculation::NamedTuple
)
    l => zeros(connectedgraph_size(cg)[1])
end

allocate_intermediate(m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit) =
    allocate_intermediate(proximity_measure(cgi), m, cgi)
# Compute for the specific proximity_measure
compute!(output::Pair, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:Union{RSP,RandomWalk}}) =
    compute_sensitivity!(output, proximity_measure(ti), m, ti)
# Finalize the output for the proximity_measure after all targets run
function finalize_output!((level, output)::Pair, m::SensitivityAnalysis, cgi::ConnectedGraphInit, intermediates)
    finalize_output!(output, proximity_measure(cgi), m, cgi, intermediates)
end


# ExpectedCost ###########################################################

# Allocate storage for ExpectedCost
function allocate_intermediate(::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit)
    (; W, C, CW, Z_full, Y_full) = precalculation(cgi)

    # kB and kΣ are set up as zeroed-out W matrices
    intermediates = (; 
        # Sparse matrices. `mapnz` means we keep the sparse structure but 
        # initialise values to zero internally
        kB = mapnz(_ -> 0.0, W),
        kΣ = mapnz(_ -> 0.0, W),
        # Diagonal vectors
        mdiagZⁱ = zeros(length(targetids(cgi))),
        mdiagC̄Zⁱ = zeros(length(targetids(cgi))),
        # Allocate Z-size matrices
        MᵀZ_full = zeros(connectedgraph_size(cgi)),
        RHS_full = zeros(connectedgraph_size(cgi)),
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
        Z_full[:, node] .= Z
        Y_full[:, node] .= Y
        intermediates.mdiagZⁱ[node] = mdiagZⁱ
        intermediates.mdiagC̄Zⁱ[node] = mdiagZⁱ * Zⁱ[idx] * Y[idx]
    end

    return intermediates
end
# For ExpectedCost sensitivity we need to loop over Z calculations twice
# Once here to build the diagonal vector that we need later
# Compute for ExpectedCost
# We have to do all compute in update_output! so we have the diagonal available
function compute_sensitivity!(
    output::Pair, ::ExpectedCost, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit
)
    node = target(ti).node

    # We already precomputed Z and Y, 
    # so add them to storage early so they arent coputed elsewhere
    (; Z_full, Y_full) = ti
    Z = readonlyarray(workspace(ti) .= view(Z_full, :, node))
    Y = readonlyarray(workspace(ti) .= view(Y_full, :, node))
    storage(ti)[:Z] = Z
    storage(ti)[:Y] = Y

    (; W, C, CW, K, qˢ, qᵗ, IW_adj_factorization, mdiagZⁱ, Zrows, MᵀZ_full, kB, kΣ, RHS_full) = ti

    X3 = workspace(ti) .= mdiagZⁱ .* Zrows
    Zⁱ = workspace(ti) .= _inv.(Z)

    Kd = workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
    Md = workspace(ti) .= qˢ .* Kd .* qᵗ
    MZⁱ = workspace(ti) .= Md .* Zⁱ
    MᵀZ = ldiv!(ti, IW_adj_factorization, (workspace(ti) .= MZⁱ)) # MᵀZ = MZⁱ' / IW
    MᵀZ_full[:, node] .= MᵀZ

    # Unrolled broadcast and matrix multiplications
    # RHS = ((M .* Zⁱ .* C̄ᵣ)' - (MᵀZ * CW) + (X3 * CW))
    C̄ᵣ = workspace(ti) .= Y .* Zⁱ
    view(RHS_full, :, node) .+= Md .* Zⁱ .* C̄ᵣ
    foreachnz(CW) do i, j, n
        if i == node
            # The contribution is the target column of X3, scaled by the value.
            for k in 1:length(X3)
                RHS_full[j, k] += X3[k] * CW.nzval[n]
            end
        end
        # MᵀZ * CW is subtracted along the row
        RHS_full[j, node] -= MᵀZ[i] .* CW.nzval[n]
    end

    return output
end
# Finalize for ExpectedCost after all targets run
function finalize_output!(
    output::AbstractArray, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit, intermediates::NamedTuple
)
    (; W, C, CW, IW_adj_factorization, A_rowsums, A, Aⁱ, Z_full, Y_full, Zrows_full) = cgi
    (; kB, kΣ, mdiagZⁱ, mdiagC̄Zⁱ, MᵀZ_full, RHS_full) = intermediates

    free!(workspaces(cgi))

    # Full matrix solve, but broken into columns to halve the memory use.
    # Reuse RHS as dest X5
    ws = workspace(cgi) 
    for j in axes(RHS_full, 2)
        rhs = view(RHS_full, :, j)
        ws .= rhs
        ldiv!(solver(cgi), rhs, IW_adj_factorization, ws)
    end
    X5_full = RHS_full

    # Use workspaces for columns
    X5 = workspace(cgi)
    X6 = workspace(cgi)

    cur_j = 0
    foreachnz(W) do i, j, n
        if j != cur_j
            cur_j = j
        end
        Z = view(Z_full, j, :)
        Y = view(Y_full, j, :)
        @views X5 .= X5_full[i, :] .- mdiagC̄Zⁱ .* Zrows_full[:, i]
        @views X6 .= MᵀZ_full[i, :] .- mdiagZⁱ .* Zrows_full[:, i]
        kB.nzval[n] = W.nzval[n] * ((Z' * X6)[1])
        kΣ.nzval[n] = W.nzval[n] * ((Z' * X5)[1] - (Y' * X6)[1] - C.nzval[n] * kB.nzval[n] / W.nzval[n])
    end

    kΣ_node = sum(kΣ, dims=2)
    foreachnz(Aⁱ) do i, j, n
        S_cost = kB.nzval[n] + theta(cgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[i] / A_rowsums[i]) - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        output[j] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
        return nothing
    end

    return output
end
# Transfer connected graph output to the final spatial grid
transfer_output!(dest::AbstractMatrix, source, ::SensitivityAnalysis, cgi::ConnectedGraphInit) =
    dest[sourceids(cgi)] .= source


# PowerMeanProximity ###########################################################

function allocate_intermediate(::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit)
    (; W) = cgi

    custom_weighted = CustomWeighted(nothing) # We dont need the weights in the allocation phase, just the type
    bet_edge_k = allocate_intermediate(EdgeBetweenness(custom_weighted), cgi) 
    bet_edge_k_output = allocate_output(ConnectedGraphLevel(), EdgeBetweenness(custom_weighted), cgi) 
    bet_node_k_output = allocate_output(ConnectedGraphLevel(), Betweenness(custom_weighted), cgi)
    intermediates = (; bet_edge_k, bet_edge_k_output, bet_node_k_output)

    return intermediates
end
# Store the output of compute into the output for each target for PowerMeanProximity
# This is part of PM_sensitivity in the original code
function compute_sensitivity!(output, pmp::PowerMeanProximity, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit)
    # Get stored Z from bet_edge_k rather than calculating it again
    node = target(ti).node

    (; qˢ, qᵗ, θ, Z) = ti

    # Calculate weights for this target
    weights = readonlyarray(workspace(ti) .= (qˢ .* ((Z ./ Z[node]) .^ θ) .* qᵗ))

    # Compute node and edge betweenness for these weights
    bet_edge = EdgeBetweenness(CustomWeighted(weights))
    bet_node = Betweenness(CustomWeighted(weights))

    ti_bet_edge = rebuild(ti; intermediates=intermediates(ti).bet_edge_k)
    ti_bet_node = rebuild(ti; intermediates=(;))
    compute!(ti.bet_edge_k_output, bet_edge, ti_bet_edge)
    compute!(ti.bet_node_k_output, bet_node, ti_bet_node)

    return output
end
# Finalize for PowerMeanProximity after all targets run
function finalize_output!(
    output::AbstractArray, ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit, intermediates
)
    (; A_rowsums, Aⁱ, A, Z_full) = cgi
    (; bet_edge_k_output, bet_node_k_output) = intermediates
    bet_edge_k, bet_node_k = bet_edge_k_output[2], bet_node_k_output[2]

    # Finalize edge betweenness
    bet_edge = EdgeBetweenness(CustomWeighted(nothing))
    finalize_output!(intermediates.bet_edge_k_output, bet_edge, cgi, intermediates.bet_edge_k)

    # This is from PM_sensitivy in the original code
    foreachnz(Aⁱ) do i, j, n
        I = sourceids(cgi)[i]
        S_cost = -bet_edge_k.nzval[n]
        S_likelihood = (bet_edge_k.nzval[n] * Aⁱ.nzval[n] - (bet_node_k[I] / A_rowsums[i]) * (A.nzval[n] > 0)) * theta(cgi)
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        x = _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
        output[j] += x
    end

    return output
end


# Shared utilities #################################

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
_maybe_scale(a, ::Elasticity, ::Union{StepLikelihood,StepCostToLikelihood}, ti, n) =
    a * steplikelihood(ti).nzval[n]
_maybe_scale(a, ::Elasticity, ::Union{StepCost,StepLikelihoodToCost}, ti, n) =
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

:onst store = Ref{NamedTuple}((;))

######################################################################################
# Sensitivity

function allocate_intermediate(
    ::SensitivityAnalysis{<:AbstractQuality}, cgi::ConnectedGraphInit
)
    return (;)
end
function compute_target(
    m::SensitivityAnalysis{<:AbstractQuality}, 
    ti::TargetInit{<:Union{RSP,RandomWalk}}
)
   (; M) = ti
    
    #= Sensitivity w.r.t. Quality is essentially just the landscape matrix, 
    i.e. proximity matrix * source and target quality.
    However for Sensitivity (as oppossed to Elasticity) we divide this
    by the source quality after summing, in finalize_connectedgraph_output! below.
    
    We cant do that at the target level because the numbers are 
    too small and cause floating point error in the sum. =#
    if metric(m) isa EigMax
        workspace(ti) .= v .* M .* w[targetnode(ti)]
    else
        M
    end
end
function finalize_connectedgraph_output!(
    output, 
    level::ConnectedGraphLevel,
    m::SensitivityAnalysis{<:AbstractQuality}, 
    cgi::ConnectedGraphInit, 
    intermediates
)
    # Divide final summed output by source quality.
    # This has better fp characteristics than summing smaller numbers.
    if sensitivitytype(m) isa Sensitivity
        output[sourceids(cgi)] ./= sourcequality(cgi)
    end

    return output
end

# Shared sequence

# GridGraph behavior: all levels have the same spatial raster output
function allocate_output(
    l::Level, ::ReturnCustom, m::SensitivityAnalysis{<:Permeability}, p::ConScapeProblem, args...
)
    allocate_output(l::Level, ReturnSpatialTargetSum(), m, p, args...)
end
# ConnectedGraph behavior
function allocate_output(
    l::Level, 
    ::ReturnCustom, 
    ::SensitivityAnalysis{<:Permeability}, 
    ::ConScapeProblem, 
    ::GridGraph, 
    cg::ConnectedGraph, 
    ::NamedTuple
)
    zeros(connectedgraph_size(cg)[1]) => l 
end

function allocate_intermediate(m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit)
    allocate_sensitivity_intermediate(proximity_measure(cgi), m, cgi)
end
# Compute for the specific proximity_measure
# TODO: error if someone tries to compute a single target? or handle that?
function compute_target!(
    output, l::Level, m::SensitivityAnalysis{<:Permeability}, ti::TargetInit{<:RSP}
)
    compute_target_sensitivity!(output, proximity_measure(ti), m, ti)
end
# Finalize the output for the proximity_measure after all targets run
function finalize_connectedgraph_output!(
    output, l::Level, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit, intermediates
)
    finalize_connectedgraph_output!(output, l, proximity_measure(cgi), m, cgi, intermediates)
end


# ExpectedCost ###########################################################

# Allocate storage for ExpectedCost
function allocate_sensitivity_intermediate(
    ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit
)
    (; W, C, CW) = precalculation(cgi)

    intermediates = (; 
        # Sparse matrices. `mapnz` means we keep the sparse structure but 
        # initialise values to zero internally
        # kB and kΣ are zeroed-out W matrices
        kB = mapnz(_ -> 0.0, W),
        kΣ = mapnz(_ -> 0.0, W),
        # Diagonal vectors
        mdiagZⁱ = zeros(ntargets(cgi)),
        mdiagC̄Zⁱ = zeros(ntargets(cgi)),
        # Allocate Z-size matrices
        MᵀZ_full = zeros(connectedgraph_size(cgi)),
        X5_full = zeros(connectedgraph_size(cgi)),
    )
    
    # Precalculate mdiagZ and mdiagC̄Z
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
        intermediates.mdiagZⁱ[idx] = mdiagZⁱ
        intermediates.mdiagC̄Zⁱ[idx] = mdiagZⁱ * Zⁱ[node] * Y[node]
    end

    return intermediates
end
# For ExpectedCost sensitivity we need to loop over Z calculations twice
# Once here to build the diagonal vector that we need later
# Compute for ExpectedCost
# We have to do all compute in update_output! so we have the diagonal available
function compute_target_sensitivity!(
    output, ::ExpectedCost, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit
)
    node = targetnode(ti)
    idx = targetconnectedgraphidx(ti)

    # We already precomputed Z and Y, 
    # so add them to storage early so they arent coputed elsewhere
    (; Z, Zⁱ, Y, W, C, CW, K, qˢ, qᵗ, IW_adj_factorization, mdiagZⁱ, Zrows_full, MᵀZ_full, kB, kΣ, X5_full) = ti

    X3 = view(workspace(ti), 1:ntargets(ti)) .= mdiagZⁱ .* view(Zrows_full, :, node)
    C̄ᵣ = workspace(ti) .= Y .* Zⁱ
    Kd = workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
    Md = workspace(ti) .= qˢ .* Kd .* qᵗ
    MZⁱ = workspace(ti) .= Md .* Zⁱ
    MᵀZ = ldiv!(ti, IW_adj_factorization, (workspace(ti) .= MZⁱ)) # MᵀZ = MZⁱ' / IW
    MᵀZ_full[:, idx] .= MᵀZ # Update the full matrix for later use

    # Here we do unrolled matrix multiplications to reduce memory use
    # This is basically duplicating the function:
    # RHS = (M .* Zⁱ .* C̄ᵣ)' - (MᵀZ * CW) + (X3 * CW)
    # First the broadcast
    view(X5_full, :, idx) .+= Md .* Zⁱ .* C̄ᵣ
    # Then both matmuls are calculated for non-zero values of CW.
    # This is hard to understand without a lot of work.
    foreachnz(CW) do i, j, n
        # First MᵀZ * CW is subtracted from the column of the current target node
        X5_full[j, idx] -= MᵀZ[i] .* CW.nzval[n]
        # Then X3 * CW is added where i is the current target node
        if i == node
            for k in eachindex(X3)
                X5_full[j, k] += X3[k] * CW.nzval[n]
            end
        end
    end

    return output
end
# Finalize for ExpectedCost after all targets run
function finalize_connectedgraph_output!(
    output::AbstractArray,
    ::ConnectedGraphLevel,
    ::ExpectedCost,
    m::SensitivityAnalysis{<:Permeability},
    cgi::ConnectedGraphInit,
    intermediates::NamedTuple
)
    (; W, C, CW, IW_adj_factorization, A_rowsums, A, Aⁱ, Z_full, Y_full, Zrows_full) = cgi
    (; kB, kΣ, mdiagZⁱ, mdiagC̄Zⁱ, MᵀZ_full, X5_full) = intermediates

    free!(workspaces(cgi))

    # Full matrix solve, but broken into columns to halve the memory use.
    ws = workspace(cgi) 
    for j in axes(X5_full, 2)
        rhs = view(X5_full, :, j)
        ws .= rhs
        ldiv!(solver(cgi), rhs, IW_adj_factorization, ws)
    end

    # Use workspaces for columns
    X5 = view(workspace(cgi), 1:ntargets(cgi))
    X6 = view(workspace(cgi), 1:ntargets(cgi))

    # Set kB an kΣ for non-zero values of W
    foreachnz(W) do i, j, n
        Z = view(Z_full, j, :)
        Y = view(Y_full, j, :)
        @views X5 .= X5_full[i, :] .- mdiagC̄Zⁱ .* Zrows_full[:, i]
        @views X6 .= MᵀZ_full[i, :] .- mdiagZⁱ .* Zrows_full[:, i]
        kBn = W.nzval[n] * ((Z' * X6)[])
        kB.nzval[n] = kBn
        kΣ.nzval[n] = W.nzval[n] * ((Z' * X5)[] - (Y' * X6)[] - (C.nzval[n] * kBn / W.nzval[n]))
    end

    kΣ_node = sum(kΣ, dims=2)

    # Calculate output from non-zero values  of W/Aⁱ/kB/kΣ
    foreachnz(kB) do i, j, n
        S_cost = kB.nzval[n] + theta(cgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[i] / A_rowsums[i]) - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        output[j] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
        return nothing
    end

    store[] = (; C, W, Zrows=Zrows_full, Z=Z_full, Y=Y_full, kB, kΣ, kΣ_node, diag=mdiagZⁱ, diagC=mdiagC̄Zⁱ, MᵀZ=MᵀZ_full, X5=X5_full)

    return output
end
# Transfer connected graph output to the final spatial grid
function transfer_to_gridgraph_output!(
    dest::AbstractArray,
    ::GridGraphLevel,
    source::AbstractArray,
    ::ConnectedGraphLevel,
    ::SensitivityAnalysis{<:Permeability},
    cgi::ConnectedGraphInit
)
    store[] = (; store[]..., source, dest) 
    dest[sourceids(cgi)] .= source
end


# PowerMeanProximity ###########################################################

function allocate_sensitivity_intermediate(
    ::PowerMeanProximity, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit
)
    (; W) = cgi

    custom_weighted = CustomWeighted(nothing) # We dont need the weights in the allocation phase, just the type
    bet_edge = allocate_intermediate(EdgeBetweenness(custom_weighted), cgi) 
    bet_edge_output = allocate_output(ConnectedGraphLevel(), EdgeBetweenness(custom_weighted), cgi) 
    bet_node_output = allocate_output(ConnectedGraphLevel(), Betweenness(custom_weighted), cgi)
    intermediates = (; bet_edge, bet_edge_output, bet_node_output)

    return intermediates
end
# Store the output of compute into the output for each target for PowerMeanProximity
# This is part of PM_sensitivity in the original code
function compute_target_sensitivity!(
    output, pmp::PowerMeanProximity, ::SensitivityAnalysis{<:Permeability}, ti::TargetInit
)
    # Get stored Z from bet_edge rather than calculating it again
    node = target(ti).node

    (; qˢ, qᵗ, θ, Z) = ti

    # Calculate weights for this target
    weights = readonlyarray(workspace(ti) .= (qˢ .* ((Z ./ Z[node]) .^ θ) .* qᵗ))

    # Compute node and edge betweenness for these weights
    bet_edge = EdgeBetweenness(CustomWeighted(weights))
    bet_node = Betweenness(CustomWeighted(weights))

    ti_edge = rebuild(ti; intermediates=intermediates(ti).bet_edge)
    ti_node = rebuild(ti; intermediates=(;))
    edge_out = ti.bet_edge_output[1]
    node_out = ti.bet_node_output[1]

    compute_target!(edge_out, ConnectedGraphLevel(), bet_edge, ti_edge)
    compute_target!(node_out, ConnectedGraphLevel(), bet_node, ti_node)

    return output
end
# Finalize for PowerMeanProximity after all targets run
function finalize_connectedgraph_output!(
    output::AbstractArray,
    ::ConnectedGraphLevel,
    ::PowerMeanProximity,
    m::SensitivityAnalysis{<:Permeability},
    cgi::ConnectedGraphInit,
    intermediates
)
    (; A_rowsums, Aⁱ, A, Z_full) = cgi
    (; bet_edge_output, bet_node_output) = intermediates
    bet_edge, bet_node = bet_edge_output[1], bet_node_output[1]

    # Finalize edge betweenness
    finalize_connectedgraph_output!(
        intermediates.bet_edge_output..., EdgeBetweenness(CustomWeighted(nothing)), cgi, intermediates.bet_edge
    )

    # This is from PM_sensitivy in the original code
    foreachnz(Aⁱ) do i, j, n
        I = sourceids(cgi)[i]
        S_cost = -bet_edge.nzval[n]
        S_likelihood = (bet_edge.nzval[n] * Aⁱ.nzval[n] - (bet_node[I] / A_rowsums[i]) * (A.nzval[n] > 0)) * theta(cgi)
        # Scaling depends on Elasticity/Sensitivity and w.r.t.
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        # Combination of cost / likelihood depends on w.r.t. 
        output[j] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
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

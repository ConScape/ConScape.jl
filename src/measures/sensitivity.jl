# Sensitivity

# A metric, but not a measure like EigMax
struct Summation end

abstract type SensitivityType end
struct Sensitivity <: SensitivityType end
struct Elasticity <: SensitivityType end

"""
    SensitivityAnalysis <: SpatialMeasure

    SensitivityAnalysis(; wrt, metric, sentitivitytype)

Compute sensitivity of all nodes. 

## Keywords

- `wrt`: Five types of node sensitivity are implemented: `Affinity()`, `Cost()`, 
    `Quality()`, `CostAndAffinity()` and `AffinityAndCost()`.
- `metric`: Two [`TopologicalMetric`](@ref)s are implemented to summarize the 
    landscape matrix either through summation ([`Summation()`](@ref)) or through eigen 
    analysis [`LandscapeEigen()`](@ref). The default is `Eigen()`.
- `type`: The results can be provided either as sensitivity w.r.t. `Sensitivity()`
    or w.r.t. `Elasticity()`, the latter are also known as elasticities. 
    The default is `Sensitivity()`

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
@kwdef struct SensitivityAnalysis{WRT<:InputType,M,ST<:SensitivityType} <: SpatialMeasure
    wrt::WRT
    metric::M = Summation()
    type::ST = Sensitivity()
end

wrt(m::SensitivityAnalysis) = m.wrt
metric(m::SensitivityAnalysis) = m.metric
sensitivitytype(m::SensitivityAnalysis) = m.type

# Quality can run at the target level
computelevel(m::SensitivityAnalysis{<:AbstractQuality}) = TargetLevel()
# Permeability runs for the whole connected graph
computelevel(m::SensitivityAnalysis{<:Permeability}) = ConnectedGraphLevel()

returntrait(::SensitivityAnalysis) = ReturnSpatialTargetSum()
returntrait(::SensitivityAnalysis{<:Quality}) = ReturnSpatialSourceAndTargetSum()
returntrait(::SensitivityAnalysis{<:SourceQuality}) = ReturnSpatialSourceSum()
returntrait(::SensitivityAnalysis{<:TargetQuality}) = ReturnSpatialTargetSum()

# Sensitivity needs full-size Z, Zrows matrices precalculated
needs_full_fundamentalmatrix(::SensitivityAnalysis, ::RSP) = true
needs_full_fundamentalrowmatrix(::SensitivityAnalysis, ::RSP) = true
# It may also need full Y
needs_full_costdistancematrix(::SensitivityAnalysis, rsp::RSP) = 
    proximity_measure(rsp) isa ExpectedCost
# And EigMax
needs_eigmax(m::SensitivityAnalysis, rsp::RSP) = metric(m) isa EigMax

# Sensitivity precursors
# We compute most of Proximity sensitivitly for all metrics,
# so the bulk of the calculations can be shared by multiple outputs
# But we need to separate them by Summation / EigMax metric
needs_sum_sensitivity_precursors(m::SensitivityAnalysis{<:Permeability}, rsp::RSP) = metric(m) isa Summation
needs_eigmax_sensitivity_precursors(m::SensitivityAnalysis{<:Permeability}, rsp::RSP) = metric(m) isa EigMax

# Vector vec_workspace requirements
# Quality with Summation returns M directly (0 vec_workspaces)
# Quality with EigMax uses 1 vec_workspace
num_vec_workspaces(m::SensitivityAnalysis{<:AbstractQuality}, ::RSP) = metric(m) isa EigMax ? 1 : 0
# Permeability precursors need vec_workspaces:
# - ExpectedCost precursors: ~7 vec_workspaces (2 persistent + 5 in target loop)
# - PowerMeanProximity precursors: uses EdgeBetweenness (4) + Betweenness (3) + weightfunc (1)
# - compute_connectedgraph! for PowerMeanProximity: 1 vec_workspace
num_vec_workspaces(::SensitivityAnalysis{<:Permeability}, ::RSP) = 8

# Sparse sp_workspace requirements
# Permeability precursors need kB and kΣ sparse matrices
num_sp_workspaces(::SensitivityAnalysis{<:Permeability}, ::RSP) = 2

# w.r.t Quality ############################################################################

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
        (v, λ, w) = ti.eigmax
        targetcol = vec_workspace(ti) .= v .* M .* w[targetnode(ti)] ./ (v' * w)
        return readonlyarray(targetcol)
    else
        return M
    end
end
function finalize_connectedgraph_output!(
    output, 
    m::SensitivityAnalysis{<:AbstractQuality}, 
    level::ConnectedGraphLevel,
    cgi::ConnectedGraphInit, 
)
    # Divide final summed output by source quality.
    # This has better fp characteristics than summing smaller numbers.
    if sensitivitytype(m) isa Sensitivity
        output[sourceids(cgi)] ./= sourcequality(cgi)
    end

    # Maybe scale by eigmax
    if metric(m) isa EigMax
        (v, λ, w) = cgi.eigmax
        output[sourceids(cgi)] ./= (v' * w)
    end

    return output
end

# w.r.t Permeability #######################################################################

function compute_connectedgraph!(
    output, 
    m::SensitivityAnalysis{<:Permeability}, 
    cgi::ConnectedGraphInit
)
    return compute_connectedgraph!(output, proximity_measure(cgi), m, cgi)
end

# ExpectedCost -----------------------------------------------------------------------------

# EigMax metric modifies qˢ and qᵗ
function _compute_eigmax_sensitivity_precursors(pm::ProximityMeasure, cgi::ConnectedGraphInit)
    (v, λ, w) = cgi.eigmax
    qˢ = cgi.qˢ .* v    
    qᵗ = cgi.qᵗ .* w[map(id -> id.node, targetids(cgi))]

    return _compute_sensitivity_precursors(pm, cgi, qˢ, qᵗ)
end

# Summation uses the standard qˢ and qᵗ
function _compute_sum_sensitivity_precursors(pm::ProximityMeasure, cgi::ConnectedGraphInit)
    (; qˢ, qᵗ) = cgi
    return _compute_sensitivity_precursors(pm, cgi, qˢ, qᵗ)
end

# These are computed first ast they can be reused for all permeability_sensitivities
function _compute_sensitivity_precursors(::ExpectedCost, cgi::ConnectedGraphInit, qˢ, qᵗ)
    (; W, C, CW, CW_t, A, Aⁱ, F_IW_adj, A_rowsums, Z_full, Y_full, Zrows_full) = cgi

    # Sparse matrices with same structure as W, initialized to zero
    kB = sp_workspace(cgi)
    kΣ = sp_workspace(cgi)
    fill!(kB.nzval, 0.0)
    fill!(kΣ.nzval, 0.0)

    # Diagonal vectors - these persist across the target loop so cannot use workspaces
    # (TargetInit constructor calls free! on vec_workspaces)
    mdiagZⁱ = zeros(ntargets(cgi))
    mdiagC̄Zⁱ = zeros(ntargets(cgi))

    # Allocate Z-size matrices
    # These can't be used split into column vectors unless we calculate them twice. 
    # This is a serious option if RAM turns out to be more limiting than CPU time.
    MᵀZ_full = mat_workspace(cgi)
    X5_full = mat_workspace(cgi)

    
    # Loop to calculate diagonals mdiagZ and mdiagC̄Z
    for target::TargetID in targetids(cgi)
        # Precalculate for this target and graph measures
        ti = TargetInit(cgi, target)
        (; K, Y, Z, Zⁱ) = ti
        dt = distance_transformation(ti)
        node = target.node
        idx = target.connectedgraphidx

        Kd = vec_workspace(ti) .= _diff_KD(dt).(K)
        Md = vec_workspace(ti) .= qˢ .* Kd .* qᵗ[idx]
        x = sum(Md) * Zⁱ[node]
        mdiagZⁱ[idx] = x
        mdiagC̄Zⁱ[idx] = x * Zⁱ[node] * Y[node]
    end


    # Loop to calcualte MᵀZ and X5
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        node = target.node
        idx = target.connectedgraphidx

        # We already precomputed Z and Y, 
        # so add them to storage early so they arent coputed elsewhere
        (; Z::RVDe, Zⁱ::RVDe, K::RVDe, Y::RVDe) = ti

        C̄ᵣ = vec_workspace(ti) .= Y .* Zⁱ
        Kd = vec_workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
        Md = vec_workspace(ti) .= qˢ .* Kd .* qᵗ[idx]
        MZⁱ = vec_workspace(ti) .= Md .* Zⁱ
        MᵀZ = ldiv!(ti, F_IW_adj, (vec_workspace(ti) .= MZⁱ)) # MᵀZ = MZⁱ' / IW
        MᵀZ_full[:, idx] .= MᵀZ # Update the full matrix for later use

        # Here we do unrolled matrix multiplications to reduce memory use.
        # This is essentially the same as:
        # RHS = (M .* Zⁱ .* C̄ᵣ)' - (MᵀZ * CW) + (X3 * CW)
        # First the broadcast
        view(X5_full, :, idx) .= Md .* Zⁱ .* C̄ᵣ
        # Then MᵀZ * CW is subtracted row by row
        matmul_by_col!(-, X5_full, MᵀZ, idx, CW)
    end
        
    X3 = view(vec_workspace(cgi), 1:ntargets(cgi))
    # Last we add X3 by source
    for node in 1:nsources(cgi)
        X3 .= mdiagZⁱ .* view(Zrows_full, :, node)
        # And X3 * CW is added column by column
        matmul_by_row!(+, X5_full, X3, node, CW_t)
    end

    rhs_copy = vec_workspace(cgi) 
    for j in axes(X5_full, 2)
        rhs = view(X5_full, :, j)
        rhs_copy .= rhs
        ldiv!(solver(cgi), rhs, F_IW_adj, rhs_copy)
    end

    # Use smaller vec_workspaces 
    X5 = view(vec_workspace(cgi), 1:ntargets(cgi))
    X6 = view(vec_workspace(cgi), 1:ntargets(cgi))

    foreachnz(W) do i, j, n
        Z = view(Z_full, j, :)
        Y = view(Y_full, j, :)
        @views X5 .= X5_full[i, :] .- mdiagC̄Zⁱ .* Zrows_full[:, i]
        @views X6 .= MᵀZ_full[i, :] .- mdiagZⁱ .* Zrows_full[:, i]
        kBn = W.nzval[n] * ((Z' * X6)[])
        kB.nzval[n] = kBn
        kΣ.nzval[n] = W.nzval[n] * ((Z' * X5)[] - (Y' * X6)[] - (C.nzval[n] * kBn / W.nzval[n]))
    end
    
    # Put back the matrix workspaces we used
    put!(mat_workspaces(cgi), X5_full)
    put!(mat_workspaces(cgi), MᵀZ_full)

    return (; kB, kΣ)
end

function compute_connectedgraph!(
    output, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit
)
    (; Aⁱ::MSp, A_rowsums::RVDe) = cgi
    (; kB::MSp, kΣ::MSp) = if metric(m) isa EigMax
        cgi.eigmax_sensitivity_precursors
    else
        cgi.sum_sensitivity_precursors
    end

    kΣ_node = sum(kΣ; dims=2)
    # Calculate output from non-zero values  of W/Aⁱ/kB/kΣ
    foreachnz(kB) do i, j, n
        J = sourceids(cgi)[j]
        S_cost = kB.nzval[n] + theta(cgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[i] / A_rowsums[i]) - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        output[J] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
        return nothing
    end

    if metric(m) isa EigMax
        (v, λ, w) = cgi.eigmax 
        vTw = (v' * w)
        output[sourceids(cgi)] ./= vTw
    end

    return output
end

# PowerMeanProximity -----------------------------------------------------------------------

function _compute_sensitivity_precursors(::PowerMeanProximity, cgi::ConnectedGraphInit, qˢ, qᵗ)
    (; W, A, Aⁱ, A_rowsums, Z_full) = cgi

    # We use a custom weigth function as this is not the standard
    # betweenness weight - usually just K or M
    function weightfunc(ti)
        (; θ, Z, Zⁱ) = ti
        node = targetnode(ti)
        idx = targetconnectedgraphidx(ti)
        return readonlyarray(vec_workspace(ti) .= (qˢ .* ((Z .* Zⁱ[node]) .^ θ) .* qᵗ[idx]))
    end
    custom_weighted = CustomWeighted(weightfunc) # We dont need the weights in the allocation phase, just the type
    finallevel = ConnectedGraphLevel()

    edge_bet = EdgeBetweenness(custom_weighted)
    node_bet = Betweenness(custom_weighted)
    edge_output = allocate_connectedgraph_output(finallevel, edge_bet, cgi).output
    node_output = allocate_connectedgraph_output(finallevel, node_bet, cgi).output

    # Compute EdgeBetweenness
    compute_connectedgraph!(edge_output, edge_bet, cgi)
    # Compute Betweenness
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        compute_target!(node_output, finallevel, node_bet, ti)
    end

    return (; edge_output, node_output)
end

function compute_connectedgraph!(
    output,
    ::PowerMeanProximity,
    m::SensitivityAnalysis{<:Permeability},
    cgi::ConnectedGraphInit
)
    (; A, Aⁱ, A_rowsums) = cgi

    (; edge_output, node_output) = if metric(m) isa EigMax
        cgi.eigmax_sensitivity_precursors
    else
        cgi.sum_sensitivity_precursors
    end

    node_sensitivity = fill!(vec_workspace(cgi), 0.0)
    # Loop over non-zero values of Aⁱ/edge_output
    foreachnz(Aⁱ) do i, j, n
        I = sourceids(cgi)[i]
        S_cost = -edge_output.nzval[n]
        S_likelihood = (edge_output.nzval[n] * Aⁱ.nzval[n] - 
                        node_output[I] / A_rowsums[i]) * (A.nzval[n] > 0) * theta(cgi)
        # Scaling depends on Elasticity/Sensitivity and w.r.t.
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        # Combination of cost / likelihood depends on w.r.t. 
        node_sensitivity[j] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
    end

    if metric(m) isa EigMax
        (v, λ, w) = cgi.eigmax 
        vTw = (v' * w)
        node_sensitivity ./= vTw
    end

    return output[sourceids(cgi)] .= node_sensitivity 
end


# Shared utilities #########################################################################
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

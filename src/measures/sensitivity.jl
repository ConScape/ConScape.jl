const store = Ref{NamedTuple}((;))

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
needs_sensitivity_precursors(::SensitivityAnalysis{<:Permeability}, rsp::RSP) = true
needs_sensitivity_precursors(::Measure, rsp::MovementMode) = false

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
       (v, λ, w) = ti.EigMax
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
)
    # Divide final summed output by source quality.
    # This has better fp characteristics than summing smaller numbers.
    if sensitivitytype(m) isa Sensitivity
        output[sourceids(cgi)] ./= sourcequality(cgi)
    end

    return output
end

# w.r.t Permeability #######################################################################

function compute_connectedgraph!(
    output, 
    m::SensitivityAnalysis{<:Permeability}, 
    cgi::ConnectedGraphInit
)
    compute_connectedgraph!(output, proximity_measure(cgi), m, cgi)
end

# ExpectedCost -----------------------------------------------------------------------------

# These are computed first ast they can be reused for all permeability_sensitivities
function _compute_sensitivity_precursors(::ExpectedCost, cgi::ConnectedGraphInit)
    (; W, C, CW, A, Aⁱ, IW_adj_factorization, A_rowsums, Z_full, Y_full, Zrows_full) = cgi

    # Sparse matrices. `mapnz` means we keep the sparse structure but 
    # initialise values to zero internally
    # kB and kΣ are zeroed-out W matrices
    kB::MSp = mapnz(_ -> 0.0, W)
    kΣ::MSp = mapnz(_ -> 0.0, W)
    # Diagonal vectors
    mdiagZⁱ::VDe = fill!(view(workspace(cgi), 1:ntargets(cgi)), 0.0)
    mdiagC̄Zⁱ::VDe = fill!(view(workspace(cgi), 1:ntargets(cgi)), 0.0)
    # Allocate Z-size matrices
    MᵀZ_full::MDe = mworkspace(cgi)
    X5_full::MDe = mworkspace(cgi)
    K_full = zeros(size(X5_full))
    
    # Loop to calculate diagonals mdiagZ and mdiagC̄Z
    for target::TargetID in targetids(cgi)
        # Precalculate for this target and graph measures
        ti = TargetInit(cgi, target)
        (; K, qˢ, qᵗ, Y, Z, Zⁱ) = ti
        dt = distance_transformation(ti)
        node = target.node
        idx = target.connectedgraphidx

        Kd = workspace(ti) .= _diff_KD(dt).(K)
        K_full[:, idx] .= Kd
        Md = workspace(ti) .= qˢ .* Kd .* qᵗ
        x = sum(Md) * Zⁱ[node]
        mdiagZⁱ[idx] = x
        mdiagC̄Zⁱ[idx] = x * Zⁱ[node] * Y[node]
    end

    # Loop to calcualte MᵀZ and X5
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        node::Int = target.node
        idx::Int = target.connectedgraphidx

        # We already precomputed Z and Y, 
        # so add them to storage early so they arent coputed elsewhere
        (; Z::RVDe, Zⁱ::RVDe, K::RVDe, Y::RVDe, qˢ, qᵗ::Float64) = ti

        X3 = view(workspace(ti), 1:ntargets(ti)) .= mdiagZⁱ .* view(Zrows_full, :, node)
        C̄ᵣ::VDe = workspace(ti) .= Y .* Zⁱ
        Kd::VDe = workspace(ti) .= _diff_KD(distance_transformation(ti)).(K)
        Md::VDe = workspace(ti) .= qˢ .* Kd .* qᵗ
        MZⁱ::VDe = workspace(ti) .= Md .* Zⁱ
        MᵀZ::VDe = ldiv!(ti, IW_adj_factorization, (workspace(ti) .= MZⁱ)) # MᵀZ = MZⁱ' / IW
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
    end

    store[] = (; C, W, Zrows=Zrows_full, Z=Z_full, Y=Y_full, K=K_full, kB, kΣ, diag=mdiagZⁱ, diagC=mdiagC̄Zⁱ, MᵀZ=MᵀZ_full, X5=X5_full)
    
    # Put back the matrix workspaces we used
    put!(mworkspaces(cgi), X5_full)
    put!(mworkspaces(cgi), MᵀZ_full)

    return (; kB, kΣ)
end

function compute_connectedgraph!(
    output, ::ExpectedCost, m::SensitivityAnalysis{<:Permeability}, cgi::ConnectedGraphInit
)
    (; Aⁱ::MSp, A_rowsums::RVDe, sensitivity_precursors) = cgi
    (; kB::MSp, kΣ::MSp) = sensitivity_precursors

    kΣ_node = sum(kΣ; dims=2)
    # Calculate output from non-zero values  of W/Aⁱ/kB/kΣ
    foreachnz(kB) do i, j, n
        S_cost = kB.nzval[n] + theta(cgi) * kΣ.nzval[n]
        S_likelihood = (kΣ_node[i] / A_rowsums[i]) - kΣ.nzval[n] * Aⁱ.nzval[n]
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        output[j] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
        return nothing
    end

    return output
end

# PowerMeanProximity -----------------------------------------------------------------------

function _compute_sensitivity_precursors(::PowerMeanProximity, cgi::ConnectedGraphInit)
    (; W, A, Aⁱ, A_rowsums, Z_full) = cgi

    # We use a custom weigth function as this is not the standard
    # betweenness weight - usually just K or M
    function weightfunc(ti)
        (; qˢ, qᵗ, θ, Z, Zⁱ) = ti
        node = targetnode(ti)
        return readonlyarray(workspace(ti) .= (qˢ .* ((Z .* Zⁱ[node]) .^ θ) .* qᵗ))
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
    (; A, Aⁱ, A_rowsums, sensitivity_precursors) = cgi
    (; edge_output, node_output) = sensitivity_precursors::NamedTuple{<:Any,Tuple{MSp,MDe}}

    # Loop over non-zero values of Aⁱ/edge_output
    foreachnz(Aⁱ) do i, j, n
        I = sourceids(cgi)[i]
        J = sourceids(cgi)[j]
        S_cost = -edge_output.nzval[n]
        S_likelihood = (edge_output.nzval[n] * Aⁱ.nzval[n] - 
                        node_output[I] / A_rowsums[i]) * (A.nzval[n] > 0) * theta(cgi)
        # Scaling depends on Elasticity/Sensitivity and w.r.t.
        S_e_likelihood_scaled = _maybe_scale(S_likelihood, sensitivitytype(m), wrt(m), cgi, n)
        S_e_cost_scaled = _maybe_scale(S_cost, sensitivitytype(m), wrt(m), cgi, n)
        # Combination of cost / likelihood depends on w.r.t. 
        output[J] += _combine_sensitivity(wrt(m), S_e_likelihood_scaled, S_e_cost_scaled, cgi, n)
    end

    return output
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

Base.@constprop :aggressive @inline function get_or_compute_target!(ti::TargetInit, m::Measure)
    store = storage(ti)
    x = Symbol(m)
    if haskey(store, x) 
        val = store[x]
        return val
    else
        val = compute_target(m, ti)
        if val isa AbstractVector
            store[x] = readonlyarray(val)
            return readonlyarray(val)
        else
            return val
        end
    end
end
Base.@constprop :aggressive @inline function get_or_compute_target!(ti::TargetInit, x::Symbol)
    store = storage(ti)
    if haskey(store, x)
        val = store[x]
        return val
    else
        val = target_precalculation!(ti, x)
        if val isa AbstractVector
            store[x] = readonlyarray(val)
            return readonlyarray(val)
        else
            return val
        end
    end
end

# Most measures dont need to deal with
# `update_connectedgraph_output!` and just define `compute_target`
function compute_target!(output, l::Level, m::Measure, ti::TargetInit)
    val = compute_target(m, ti)
    update_connectedgraph_output!(output, l, m, ti, val)
    return output
end

@generated Base.Symbol(m::Measure) = QuoteNode(nameof(m))

function num_mat_workspaces(problem::ConScapeProblem)
    mes = measures(problem)
    mov = movement(problem)

    max_workspaces =
        # These vec_workspaces need to persist between multiple measures.
        # They will not be returned, so we sum them.
        anymeasure(needs_full_fundamentalmatrix, mes, mov) +
        anymeasure(needs_full_fundamentalrowmatrix, mes, mov) +
        anymeasure(needs_full_costdistancematrix, mes, mov) +
        # These vec_workspaces are ephemeral and `put!` back within
        # the functions that use them, so we take the maximum.
        max(
            anymeasure(needs_edgebetweenness_workspace, mes, mov),
            2 * anymeasure(needs_eigmax, mes, mov),
            2 * anymeasure(needs_sum_sensitivity_precursors, mes, mov),
            2 * anymeasure(needs_eigmax_sensitivity_precursors, mes, mov),
        )

    return max_workspaces
end

function num_vec_workspaces(problem::ConScapeProblem)
    mes = measures(problem)
    mov = movement(problem)
    # Vector vec_workspaces accumulate during computation and are only freed
    # at the end with free!(), so we sum across all measures.
    # Base of 10 covers target_precalculation needs:
    # Z (2), Zⁱ (1), Y (1), K (2 with proximity measure), M (1), plus buffer
    base = 10
    return base + sum(m -> num_vec_workspaces(m, mov), values(mes); init=0)
end

function num_sp_workspaces(problem::ConScapeProblem)
    mov = movement(problem)
    mes = measures(problem)
    base = num_sp_workspaces_base(mov, solver(problem))
    return base + sum(m -> num_sp_workspaces(m, mov), values(mes); init=0)
end

# Define the default output Level for that initialisation object
defaultfinallevel(::GridGraphInit) = GridGraphLevel()
defaultfinallevel(::ConnectedGraphInit) = ConnectedGraphLevel()
defaultfinallevel(::TargetInit) = TargetLevel()

#################################################################################3
# On demand target-level variable calculation

@inline function target_precalculation!(
    ti::TargetInit{<:RSP}, x::Symbol
)
    pre = precalculation(ti)

    # Or compute and store
    output = if x === :Z
        if hasproperty(pre, :Z_full)
            _copycol(pre.Z_full, ti)
        else
            _fundamentalmatrixcol(ti)
        end
    elseif x === :Zⁱ
        _inversefundamentalmatrixcol(ti)
    elseif x === :Zrows
        if hasproperty(pre, :Zrows_full)
            _copyrow(pre.Zrows_full, ti)
        else
            _fundamentalrowmatrixrow(ti)
        end
    elseif x === :Y
        if hasproperty(pre, :Y_full)
            _copycol(pre.Y_full, ti)
        else
            _costdistancematrixcol(ti)
        end
    elseif x === :Q
        _qualitymatrixcol(ti)
    elseif x === :K
        _proximitymatrixcol(ti)
    elseif x === :M
        _landscapematrixcol(ti)
    else
        error("Unknown property $x")
    end

    return output
end
@inline function target_precalculation!(ti::TargetInit{<:RandomWalk}, x::Symbol)
    pre = precalculation(ti)

    # Or compute and store
    output = if x === :Z
        if hasproperty(pre, :Z_full)
            _copycol(pre.Z_full, ti)
        else
            _fundamentalmatrixcol(ti)
        end
    elseif x === :Zⁱ
        _inversefundamentalmatrixcol(ti)
    elseif x === :Zrows
        Zrows = ((I - W')\Matrix(sparse(targetnodes,
                                       1:length(targetnodes),
                                       1.0,
                                       size(W, 1),
                                       length(targetnodes))))'
    elseif x === :F_IW
        # For RandomWalk these have to be updated per-target
        # using Woodbury matrices, rather than being defined
        # only at the ConnectedGraph level.
        _woodburysubtochasticmatrix(ti)
    elseif x === :F_IW_adj
        ti.F_IW'
    elseif x === :W
        # TODO do with less allcations at the target level
        W = copy(ti.P)
        W[target(ti).node, :] .= 0 # set target node as killing (t row set to 0)
        W
    elseif x === :CW
        stepcost(ti) .* ti.W
    elseif x === :K
        _proximitymatrixcol(ti)
    elseif x === :M
        _landscapematrixcol(ti)
    elseif x === :Q
        _qualitymatrixcol(ti)
    elseif x === :Y
        if hasproperty(pre, :Y_full)
            _copycol(pre.Y_full, ti)
        else
            _costdistancematrixcol(ti)
        end
    else
        error("Unknown property $x")
    end

    return output
end
@inline function target_precalculation!(ti::TargetInit{<:LCP}, x::Symbol)
    # Either retrieve from storage, or calculate and store
    output = if x == :shortest_paths
        # TODO: this is very slow, use Eikonal.jl instead
        return Graphs.dijkstra_shortest_paths(ti.cost_weighted_digraph, target(ti).node)::Graphs.DijkstraState{Float64,Int}
    elseif x == :K # "proximity vector"
        (; shortest_paths) = ti
        # TODO this should error earlier
        readonlyarray(vec_workspace(ti) .= distance_transformation(ti).(shortest_paths.dists))
    elseif x === :M # "landscape vector"
        _landscapematrixcol(ti)
    elseif x === :Q
        _qualitymatrixcol(ti)
    else
        error("Unknown property $x")
    end

    return output
end

###########################################################################################
# Variable generation for TargetInit

# Copy a column  from a precalculated full matrix
_copyrow(A, ti) = readonlyarray(vec_workspace(ti) .= view(A, targetconnectedgraphidx(ti), :))
_copycol(A, ti) = readonlyarray(vec_workspace(ti) .= view(A, :, targetconnectedgraphidx(ti)))

function _proximitymatrixcol(ti::TargetInit{<:Union{RSP,RandomWalk}})::RVDe
    pm = proximity_measure(ti)
    dt = distance_transformation(ti)
    distances = get_or_compute_target!(ti, pm)
    proximities = if pm isa DistanceMeasure && !isnothing(dt)
        vec_workspace(ti) .= dt.(distances)
    else
        vec_workspace(ti) .= distances
    end
    _maybe_set_diagonal!(proximities, ti)

    return readonlyarray(proximities)
end

function _fundamentalmatrixcol(ti::TargetInit{<:Union{RSP,RandomWalk}})
    # `Z = (I - W) \ i` where b is column of the a diagonal matrix of 1s
    # Solving `(I - W) * Z = i` for unknown `Z`
    i = _identity_col!(vec_workspace(ti), target(ti))
    i_copy = _identity_col!(vec_workspace(ti), target(ti))
    Z = ldiv!(solver(ti), i, ti.F_IW, i_copy)
    # Clamp Z to minimum positive value to prevent underflow to zero.
    # At high theta, exp(-θ*C) can underflow, causing Z to become exactly 0.
    # This causes NaN when computing Z * Zⁱ (since 0 * floatmax = NaN).
    # By clamping to floatmin, we get Z * Zⁱ = floatmin * floatmax = 1.0.
    Z .= max.(Z, floatmin(eltype(Z)))
    return readonlyarray(Z)
end

function _inversefundamentalmatrixcol(ti)
    readonlyarray(_inv!(vec_workspace(ti), ti.Z))
end

function _fundamentalrowmatrixrow(ti::TargetInit{<:Union{RSP,RandomWalk}})
    Zrows = if _issquare(ti)
        ti.Z
    else
        b = _identity_col!(vec_workspace(ti), target(ti))
        Zrows = ldiv!(ti, ti.F_IW_adj, b)
        readonlyarray(Zrows)
    end

    return Zrows
end

# TODO: is this the most correct name for Y ?
function _costdistancematrixcol(ti)
    (; CW, Z, F_IW) = ti
    # Solve: (I - W) \ (C .* W) * Z ./ Z
    # Manual matmul is *much* faster with sparse/dense.
    # Otherwise this is 99% of the run time.
    RHS = fill!(vec_workspace(ti), 0.0)
    foreachnz(CW) do i, j, n
        RHS[i] += CW.nzval[n] * Z[j] 
    end

    Y = ldiv!(ti, F_IW, RHS)

    return readonlyarray(Y)
end

function _landscapematrixcol(ti::TargetInit)
    (; qˢ, K, qᵗ) = ti
    return readonlyarray(vec_workspace(ti) .= qˢ .* K .* qᵗ)
end

function _qualitymatrixcol(ti::TargetInit)
    (; qˢ, qᵗ) = ti
    return readonlyarray(vec_workspace(ti) .= qˢ .* qᵗ)
end

"""
    _woodburysubtochasticmatrix(ti::TargetInit{<:RandomWalk})

Construct a Woodbury matrix representing (I - W_t)^{-1} for RandomWalk target t.

# Numerical Stability Strategy

For RandomWalk, we need (I - W_t)^{-1} where W_t is P with row t zeroed (substochastic).

**Problem with naive approach**: Using I-P as the base matrix fails because P is stochastic
(rows sum to 1), making I-P singular. LU factorization has near-zero pivots (~1e-16), causing
catastrophic numerical errors in the Woodbury formula (Z values become 0 instead of ~1).

**Solution**: Use I-W₁ (substochastic, first target row zeroed) as the base. Since W₁ has
spectral radius < 1, I-W₁ is well-conditioned and invertible.

For target t ≠ base:
- IW_t = IW_base + ΔW where ΔW = W_base - W_t
- ΔW has row base = -P[base,:] (restore this row) and row t = P[t,:] (zero this row)
- This is a rank-2 update: ΔW = U * V where U is n×2 and V is 2×n

For target t == base:
- IW_t = IW_base directly (no update needed)
"""
function _woodburysubtochasticmatrix(ti::TargetInit{<:RandomWalk})
    (; P, F_IW_base, base_target_node) = ti
    t = target(ti).node
    n = LinearAlgebra.checksquare(P)

    # If this is the base target, just return the base factorization directly
    if t == base_target_node
        return F_IW_base
    end

    # Rank-2 Woodbury update: IW_t = IW_base + U * C * V
    #
    # IW_base has row base_target_node = identity row (zeroed in W)
    # IW_t needs row base_target_node = IP[base,:] and row t = identity row
    #
    # The difference ΔW = IW_t - IW_base:
    #   Row base: IW_t[base,:] - IW_base[base,:] = (I-P)[base,:] - e_base = -P[base,:]
    #   Row t:    IW_t[t,:] - IW_base[t,:] = e_t - (I-P)[t,:] = P[t,:]
    #
    # So ΔW = U * V where:
    #   U = [e_base | e_t]  (n×2 matrix with unit vectors as columns)
    #   V = [-P[base,:]; P[t,:]]  (2×n matrix)

    U = zeros(n, 2)
    V = zeros(2, n)
    C = Matrix{Float64}(I, 2, 2)  # 2×2 identity

    U[base_target_node, 1] = 1.0  # e_base in column 1
    U[t, 2] = 1.0                  # e_t in column 2

    V[1, :] .= .-view(P, base_target_node, :)  # -P[base,:]
    V[2, :] .= view(P, t, :)                    # P[t,:]

    return Woodbury(F_IW_base, U, C, V)
end

# Custom `inv` broadcast that avoids Inf
_inv(Z::AbstractArray) = _inv!(similar(Z), Z)
_inv!(Zⁱ::AbstractArray, Z::AbstractArray) = broadcast!(_inv, Zⁱ, Z)
function _inv(x::T) where T<:Number 
    i = inv(x)
    return isfinite(i) ? i : floatmax(T)
end


# Fill a vector with zeros, and one for the target node
# If this column it was part of a square matrix it would be an identity matrix
function _identity_col!(vec_workspace, target::TargetID)
    fill!(vec_workspace, 0.0)
    vec_workspace[target.node] = 1.0
    return vec_workspace
end


# What are these, how are they different to the RSP versions?
# compute_target(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute_target(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::FunctionalHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * sourcequality_spatial(init)[target] * targetquality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0

# function _check_z(ti::TargetInit{<:RSP})
#     # Check that values in Z are not too small
#     # TODO: does this make sense for single targets
#     if check(ti) && minimum(ti.Z) * minimum(nonzeros(ti.CW)) == 0
#         @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
#     end
# end

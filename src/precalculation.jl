#################################################################################3
# Connected graph level variable precalculation

# needs_full_ traits indicates that the full matrix is needed in the 
# `finalize_output!` stage. Before then it will not be correct.
# TODO: should it error to use these outside of finalize_output! ?
needs_full_fundamentalmatrix(::Measure, ::MovementMode) = false
needs_full_fundamentalmatrix(::EdgeBetweenness, ::RSP) = true
needs_full_fundamentalmatrix(::SensitivityAnalysis, ::RSP) = true

needs_full_fundamentalrowmatrix(::Measure, ::MovementMode) = false
needs_full_fundamentalrowmatrix(::EdgeBetweenness, ::RSP) = true
needs_full_fundamentalrowmatrix(::SensitivityAnalysis, ::RSP) = true

needs_full_costdistancematrix(::Measure, ::MovementMode) = false
needs_full_costdistancematrix(::SensitivityAnalysis, rsp::RSP) = 
    proximity_measure(rsp) isa ExpectedCost

anymeasure(f, measures::NamedTuple, mov::MovementMode) = 
    anymeasure(f, values(measures), mov)
anymeasure(f, measures::Tuple{Vararg{Measure}}, mov::MovementMode) = 
    any(map(m -> f(m, mov), measures)) 

# TODO: Move these to a precalculation.jl file with the computation precalculations
function sparse_precalculation(problem::ConScapeProblem{<:RSP}, graph::ConnectedGraph)
    _check_inputs(problem, graph)
    P, A_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    W = _substochasticmatrix(movement(problem), P, stepcost(graph)::AbstractMatrix)
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    A = steplikelihood(graph)::AbstractMatrix
    Aⁱ = mapnz(inv, A)
    if solver(problem) isa VectorSolver
        IW_adj = IW'
        # Use adjoint factorization of A rather than recalculating for A'
        IW_adj_factorization = IW_factorization'
    else # LinearSolver
        # LinearSolve.jl cant handle the adjoint 
        # so we duplicate work and allocations
        IW_adj = sparse(IW')
        IW_adj_factorization = init(solver(problem), IW_adj)
    end
    C = stepcost(graph)
    CW = C .* W

    # For completeness we also move these to precalculations
    θ = theta(problem)
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    return (; P, W, IW, IW_adj, C, CW, IW_factorization, IW_adj_factorization, A, Aⁱ, A_rowsums, θ, qᵗ, qˢ)
end
function sparse_precalculation(p::ConScapeProblem{<:LCP}, graph::ConnectedGraph)
    _check_inputs(p, graph)
    P, L_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    # TODO: use a raster based shortest path algorithm from Geomorphometry.jl
    # disjkstra is especially slow due to allocations,
    # searchsorted for index lookups, and Dict getindex/setindex!.
    cost_weighted_digraph = SimpleWeightedDiGraph(stepcost(graph)::AbstractMatrix)
    dsp1 = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, 1)
    parents = dsp1.parents
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in 1:length(parents)]

    # For completeness we also move these to precalculations
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    (; P, L_rowsums, cost_weighted_digraph, path_allocs, qᵗ, qˢ)
end
function sparse_precalculation(problem::ConScapeProblem{<:RandomWalk}, graph::ConnectedGraph)
    _check_inputs(problem, graph)
    P, L_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    Lⁱ = mapnz(inv, steplikelihood(graph)::AbstractMatrix)
    PC = P .* stepcost(graph)::AbstractMatrix
    PC_rowsums = sum(PC; dims=2)
    IP = I - P
    IP_factorization = init(solver(problem), IP)

    # For completeness we also move these to precalculations
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    return (; Lⁱ, L_rowsums, P, PC, PC_rowsums, IP, IP_factorization, qᵗ, qˢ)
end
function sparse_precalculation(::ConScapeProblem{<:Euclidean}, ::ConnectedGraph)
    (;)
end

# As much as possible, we avoid dense matrix precalculation as
# they are gigabyte each for a 600 * 600 matrix with all targets used.
# And 8 GB for 1000 * 1000 !
# But when its unnavoidable, this is where it happens.
# TODO: preallocate these from the WindowProblem level
function dense_precalculation(cgi::ConnectedGraphInit{<:RSP})
    # Generate a full size Z and Zrows where needed, 
    # by looping over targetids and triggering `compute` calling `ti.Z` and/or `ti.Zrows`.
    mov = movement(cgi)
    mes = measures(cgi)

    function needs_full(args...)
        needs_full_fundamentalmatrix(args...) || 
        needs_full_fundamentalrowmatrix(args...) ||
        needs_full_costdistancematrix(args...)
    end

    if anymeasure(needs_full, mes, movement(cgi))
        Z_full = if anymeasure(needs_full_fundamentalmatrix, mes, mov)
            Matrix{Float64}(undef, connectedgraph_size(cgi))
        end
        Zrows_full = if anymeasure(needs_full_fundamentalrowmatrix, mes, mov)
            Matrix{Float64}(undef, connectedgraph_size(cgi))
        end
        Y_full = if anymeasure(needs_full_costdistancematrix, mes, mov)
            Matrix{Float64}(undef, connectedgraph_size(cgi))
        end
        merge(
            isnothing(Z_full) ? (;) : (; Z_full), 
            isnothing(Zrows_full) ? (;) : (; Zrows_full),
            isnothing(Y_full) ? (;) : (; Y_full),
        )
    else
        (;)
    end
end
function dense_precalculation(::ConnectedGraphInit)
    (;)
end

# Variable generation for ConnectedGraphInit
function _transitionprobability(L::SparseMatrixCSC)
    source_sums = vec(sum(L, dims=2))
    source_scaling = inv.(source_sums)
    P = Diagonal(source_scaling) * L
    return P, source_sums
end
# Connectedstochastic
function _substochasticmatrix(rsp::RSP, P::SparseMatrixCSC, C::SparseMatrixCSC)
    @assert LinearAlgebra.checksquare(C) == LinearAlgebra.checksquare(P)
    W = P .* exp.(-theta(rsp) .* C)
    replace!(W.nzval, NaN => 0.0)
    return W
end

# function _check_z(ti::TargetInit{<:RSP})
#     # Check that values in Z are not too small
#     # TODO: does this make sense for single targets
#     if check(ti) && minimum(ti.Z) * minimum(nonzeros(ti.CW)) == 0
#         @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
#     end
# end

# Computes the stationary distribution of a random walk following the transition probability matrix
function _stationary_distribution(solver::Solver, P::SparseMatrixCSC)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI_factorization = init(solver, PI)
    PI[1, :] .= 1
    v = zeros(n)
    v1 = zeros(n)
    v[1] = v1[1] = 1
    return ldiv!(solver, v, PI_factorization, v1)
end


#################################################################################3
# Target level variable precalculation

@inline function target_precalculation!(ti::TargetInit{<:RSP}, x::Symbol)::ReadOnlyArray
    st = storage(ti)
    # Either retrieve from storage
    haskey(st, x) && return st[x]

    # Or compute and store
    st[x] = output = if x === :Z
        _fundamentalmatrix(ti)
    elseif x === :Zⁱ
        _inversefundamentalmatrix(ti)
    elseif x === :Zrows
        _fundamentalrowmatrix(ti)
    elseif x === :Y
        _costdistancematrix(ti)
    elseif x === :Q
        _qualitymatrix(ti)
    elseif x === :K
        _proximitymatrix(ti)
    elseif x === :M
        _landscapematrix(ti)
    else
        error("Unknown property $x")
    end

    return output
end
@inline function target_precalculation!(ti::TargetInit{<:RandomWalk}, x::Symbol)
    st = storage(ti)

    # Either retrieve from storage
    haskey(st, x) && return st[x]

    # Or compute and store
    st[x] = output = if x === :Z
        _fundamentalmatrix(ti)
    elseif x === :Zⁱ
        _inversefundamentalmatrix(ti)
    elseif x === :Zrows
        _fundamentalrowmatrix(ti)
    elseif x === :IW_factorization
        # For RandomWalk these have to be updated per-target
        # using Woodbury matrices, rather than being defined
        # only at the ConnectedGraph level.
        _woodburysubtochasticmatrix(ti)
    elseif x === :IW_adj_factorization
        ti.IW_factorization'
    elseif x === :W
        # TODO do with less allcations at the target level
        W = copy(ti.P)
        W[target(ti).node, :] .= 0 # set target node as killing (t row set to 0)
        W
    elseif x === :CW
        stepcost(ti) .* ti.W
    elseif x === :K
        _proximitymatrix(ti)
    elseif x === :M
        _landscapematrix(ti)
    elseif x === :Q
        _qualitymatrix(ti)
    elseif x === :Y
        _costdistancematrix(ti)
    else
        error("Unknown property $x")
    end

    return output
end
@inline function target_precalculation!(ti::TargetInit{<:LCP}, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]
    # Either retrieve from storage, or calculate and store
    output = if x == :shortest_paths
        # TODO: this is very slow, use Eikonal.jl instead
        return Graphs.dijkstra_shortest_paths(ti.cost_weighted_digraph, target(ti).node)::Graphs.DijkstraState{Float64,Int}
    elseif x == :K # "proximity vector"
        (; shortest_paths) = ti
        # TODO this should error earlier
        readonlyarray(workspace(ti) .= distance_transformation(ti).(shortest_paths.dists))
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
            workspace(ti) .= dt.(distances)
        else
            distances
        end
    else
        distances
    end
    proximities = _maybe_set_diagonal!(ti, proximities)
    return readonlyarray(proximities)
end
function _fundamentalmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    b = _diag_vec!(workspace(ti), target(ti))
    b_copy = _diag_vec!(workspace(ti), target(ti))
    Z = ldiv!(solver(ti), b, ti.IW_factorization, b_copy)
    if hasproperty(precalculation(ti), :Z_full)
        precalculation(ti).Z_full[:, target(ti).node] .= Z
    end
    return readonlyarray(Z)
end
function _inversefundamentalmatrix(ti)
    readonlyarray(_inv!(workspace(ti), ti.Z))
end
function _fundamentalrowmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    Zrows = if connectedgraph_size(ti)[1] != connectedgraph_size(ti)[2]
        b = _diag_vec!(workspace(ti), target(ti))
        Zrows = ldiv!(ti, ti.IW_adj_factorization, b)
        readonlyarray(Zrows)
    else
        ti.Z
    end
    if hasproperty(precalculation(ti), :Zrows_full)
        precalculation(ti).Zrows_full[:, target(ti).node] .= Zrows
    end

    return Zrows
end
function _costdistancematrix(ti)
    (; CW, Z, IW_factorization) = ti
    # Solve: (I - W) \ (C .* W) * Z ./ Z
    # Manual matmul is *much* faster with sparse/dense.
    # Otherwise this is 99% of the run time.
    RHS = fill!(workspace(ti), 0.0)
    foreachnz(CW) do i, j, n
        RHS[i] += CW.nzval[n] * Z[j] 
    end
    Y = ldiv!(ti, IW_factorization, RHS)
    if hasproperty(precalculation(ti), :Y_full)
        precalculation(ti).Y_full[:, target(ti).node] .= Y
    end
    return readonlyarray(Y)
end
function _landscapematrix(ti::TargetInit)
    (; qˢ, K, qᵗ) = ti
    return readonlyarray(workspace(ti) .= qˢ .* K .* qᵗ)
end
function _qualitymatrix(ti::TargetInit)
    (; qˢ, qᵗ) = ti
    return readonlyarray(workspace(ti) .= qˢ .* qᵗ)
end
function _woodburysubtochasticmatrix(ti::TargetInit{<:RandomWalk})
    (; P, IP, IP_factorization) = ti
    t = target(ti).node
    n = LinearAlgebra.checksquare(IP)

    # Prepare a Woodbury matrix to cheaply zero out row t, without factorization
    U = fill!(reshape(workspace(ti), (n, 1)), 0.0)
    V = fill!(reshape(workspace(ti), (1, n)), 0.0)
    U[t] = 1             # Identity
    V .= .- (IP[t:t, :]) # So that IP[t, :] + UCV[t, :] .== 0
    V[t] = -P[t, t]      # So that IP[t, t] + UCV[t, t] = 1
    C = 1 # Identity

    # @assert IP + U * C * V .- (I - W)
    return Woodbury(IP_factorization, U, C, V)
end

# Custom `inv` broadcast that avoids Inf
_inv(Z::AbstractArray) = _inv!(similar(Z), Z)
_inv!(Zⁱ::AbstractArray, Z::AbstractArray) = broadcast!(_inv, Zⁱ, Z)
function _inv(x::T) where T<:Number 
    i = inv(x)
    return isfinite(i) ? i : floatmax(T)
end



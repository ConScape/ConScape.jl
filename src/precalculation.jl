#################################################################################3
# Graph level variable precalculation

# TODO: Move these to a precalculation.jl file with the computation precalculations
function connectedgraph_precalculation(problem::ConScapeProblem{<:RSP}, graph::ConnectedGraph)
    _check_inputs(problem, graph)
    P, A_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    W = _substochasticmatrix(movement(problem), P, stepcost(graph)::AbstractMatrix)
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    Aⁱ = mapnz(inv, steplikelihood(graph)::AbstractMatrix)
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
    CW = stepcost(graph) .* W

    return (; P, W, IW, IW_adj, CW, IW_factorization, IW_adj_factorization, Aⁱ, A_rowsums)
end
function connectedgraph_precalculation(p::ConScapeProblem{<:LCP}, graph::ConnectedGraph)
    _check_inputs(p, graph)
    P, L_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    # TODO: use a raster based shortest path algorithm from Geomorphometry.jl
    # disjkstra is especially slow due to allocations,
    # searchsorted for index lookups, and Dict getindex/setindex!.
    cost_weighted_digraph = SimpleWeightedDiGraph(stepcost(graph)::AbstractMatrix)
    dsp1 = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, 1)
    parents = dsp1.parents
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in 1:length(parents)]
    (; P, L_rowsums, cost_weighted_digraph, path_allocs)
end
function connectedgraph_precalculation(problem::ConScapeProblem{<:RandomWalk}, graph::ConnectedGraph)
    _check_inputs(problem, graph)
    P, L_rowsums = _transitionprobability(steplikelihood(graph)::AbstractMatrix)
    Lⁱ = mapnz(inv, steplikelihood(graph)::AbstractMatrix)
    PC = P .* stepcost(graph)::AbstractMatrix
    PC_rowsums = sum(PC; dims=2)
    IP = I - P
    IP_factorization = init(solver(problem), IP)
    return (; Lⁱ, L_rowsums, P, PC, PC_rowsums, IP, IP_factorization)
end
function connectedgraph_precalculation(::ConScapeProblem{<:Euclidean}, ::ConnectedGraph)
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
# Graph level variable precalculation

@inline function target_precalculation!(ti::TargetInit{<:RSP}, x::Symbol)::Vector{Float64}
    st = storage(ti)
    haskey(st, x) && return st[x]
    output = if x === :Z # "fundamental matrix"
        _fundamentalmatrix(ti)
    elseif x === :Zⁱ # elementwise inverse of Z
        readonlyarray(_inv!(ti.workspace, ti.Z))
    elseif x === :Zrows
        _fundamentalrowmatrix(ti)
    elseif x === :Y
        (; CW, Z, IW_factorization, workspace) = ti
        # Solve: (I - W) \ (C .* W) * Z ./ Z
        readonlyarray(ldiv!(ti, IW_factorization, mul!(workspace, CW, Z)))
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
@inline function target_precalculation!(ti::TargetInit{<:RandomWalk}, x::Symbol)
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
        W = copy(ti.P)
        t = target(ti).node
        W[t, :] .= 0 # set target node as killing (t row set to 0)
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
        readonlyarray(ldiv!(ti, IW_factorization, mul!(workspace, CW, Z)))
    else
        error("Unknown property $x")
    end
    st[x] = output
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
        (; shortest_paths, workspace) = ti
        # TODO this should error earlier
        readonlyarray(workspace .= distance_transformation(ti).(shortest_paths.dists))
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
    return readonlyarray(proximities)
end
function _fundamentalmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    workspace1, workspace2 = workspaces(ti)
    b = _diag_vec!(workspace1, target(ti))
    b_copy = _diag_vec!(workspace2, target(ti))
    Z = ldiv!(solver(ti), b, ti.IW_factorization, b_copy)
    return readonlyarray(Z)
end
function _fundamentalrowmatrix(ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; Z) = ti
    if size(Z, 1) != length(targetids(ti))
        b = _diag_vec!(ti.workspace, target(ti))
        return readonlyarray(ldiv!(ti, ti.IW_adj_factorization, b))
    else
        return readonlyarray(Z)
    end
end
function _landscapematrix(ti::TargetInit)
    (; qˢ, K, qᵗ, workspace) = ti
    return readonlyarray(workspace .= qˢ .* K .* qᵗ)
end
function _qualitymatrix(ti::TargetInit)
    (; qˢ, qᵗ, workspace) = ti
    return readonlyarray(workspace .= qˢ .* qᵗ)
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
    end |> readonlyarray
end



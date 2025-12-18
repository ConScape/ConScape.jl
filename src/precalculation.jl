
const MSp = SparseMatrixCSC{Float64,Int64}
const MDe = Matrix{Float64}
const VDe = Vector{Float64}
const RVDe = ReadOnlyArrays.ReadOnlyVector{Float64,Vector{Float64}}

#################################################################################3
# Connected graph level variable precalculation

# needs_full_ traits indicates that the full matrix is needed in the 
# `finalize_connectedgraph_output!` stage. Before then it will not be correct.
# TODO: should it error to use these outside of finalize_connectedgraph_output! ?
needs_full_fundamentalmatrix(::Measure, ::MovementMode) = false
needs_full_fundamentalrowmatrix(::Measure, ::MovementMode) = false
needs_full_costdistancematrix(::Measure, ::MovementMode) = false
needs_sum_sensitivity_precursors(::Measure, ::MovementMode) = false
needs_eigmax_sensitivity_precursors(::Measure, ::MovementMode) = false
needs_eigmax(::Measure, ::MovementMode) = false

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
    CW_t = sparse(transpose(CW))

    # For completeness we also move these to precalculations
    θ = theta(problem)
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    return (; P, W, IW, IW_adj, C, CW, CW_t, IW_factorization, IW_adj_factorization, A, Aⁱ, A_rowsums, θ, qᵗ, qˢ)
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
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in eachindex(parents)]

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

# dense_precalculation
# As much as possible, we avoid dense matrix precalculation as
# they are gigabyte each for a 600 * 600 matrix with all targets used.
# 8 GB for 1000 * 1000 !
# But when its unnavoidable, this is where it happens.
function dense_precalculation(cgi::ConnectedGraphInit{<:Union{<:RSP,<:RandomWalk}})
    # Generate a full size Z and Zrows where needed, 
    # by looping over targetids and triggering `compute_target` calling `ti.Z` and/or `ti.Zrows`.
    mov = movement(cgi)
    mes = measures(cgi)

    # Define a function to check if any measure needs a full size matrix
    function needs_full_matrix(args...)
        needs_full_fundamentalmatrix(args...) || 
        needs_full_fundamentalrowmatrix(args...) ||
        needs_full_costdistancematrix(args...) ||
        needs_eigmax(args...) ||
        needs_sum_sensitivity_precursors(args...) ||
        needs_eigmax_sensitivity_precursors(args...) ||
        needs_eigmax(args...)
    end

    # If needed, allocated and precalculate some full size matrices
    if anymeasure(needs_full_matrix, mes, movement(cgi))
        # Allocate required matrices
        Z_full = if anymeasure(needs_full_fundamentalmatrix, mes, mov)
            mworkspace(cgi) 
        end
        Zrows_full = if anymeasure(needs_full_fundamentalrowmatrix, mes, mov)
            _issquare(cgi) ? Z_full : transpose(mworkspace(cgi))
        end
        Y_full = if anymeasure(needs_full_costdistancematrix, mes, mov)
            mworkspace(cgi) 
        end

        # Precalculate required matrices column by column
        for target in targetids(cgi)
            ti = TargetInit(cgi, target)
            t = target.connectedgraphidx

            if !isnothing(Z_full)
                # Zrows (square) and Y need Z so we store it
                Z = _fundamentalmatrixcol(ti)
                storage(ti)[:Z] = Z
                Z_full[:, t] .= Z
            end
            if !isnothing(Zrows_full) && !_issquare(cgi)
                Zrows_full[t, :] .= _fundamentalrowmatrixrow(ti)
            end
            if !isnothing(Y_full)
                Y_full[:, t] .= _costdistancematrixcol(ti)
            end
        end

        precalculation = merge(
            ConScape.precalculation(cgi),  
            isnothing(Z_full) ? (;) : (; Z_full), 
            isnothing(Zrows_full) ? (;) : (; Zrows_full),
            isnothing(Y_full) ? (;) : (; Y_full),
            isnothing(Y_full) ? (;) : (; Y_full),
        )

        # EigMax
        if anymeasure(needs_eigmax, mes, mov)
            cgi_part_precalc = ConnectedGraphInit(
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), workspaces(cgi),
                mworkspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
            )
            em = _get_eigmax(mes)::EigMax
            eigmax = _compute_eigmax(em, cgi_part_precalc)
            precalculation = (; precalculation..., eigmax)
        end

        # EigMax sensitivity precursors
        if anymeasure(needs_sum_sensitivity_precursors, mes, mov)
            cgi_part_precalc = ConnectedGraphInit(
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), workspaces(cgi), 
                mworkspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
            )
            sum_sensitivity_precursors = _compute_sum_sensitivity_precursors(proximity_measure(cgi), cgi_part_precalc)
            precalculation = (; precalculation..., sum_sensitivity_precursors)
        end

        # Summation sensitivity precursors
        if anymeasure(needs_eigmax_sensitivity_precursors, mes, mov)
            cgi_part_precalc = ConnectedGraphInit(
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), workspaces(cgi), 
                mworkspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
            )
            eigmax_sensitivity_precursors = _compute_eigmax_sensitivity_precursors(proximity_measure(cgi), cgi_part_precalc)
            precalculation = (; precalculation..., eigmax_sensitivity_precursors)
        end

        return precalculation

        # Return only the required dense matrices in a NamedTuple
    else
        return (;)
    end
end
# Otherwise no dense precalculation
function dense_precalculation(::ConnectedGraphInit)
    (;)
end

function _get_eigmax(measures::NamedTuple)::EigMax
    all_eigmax = map(_get_eigmax, values(measures))
    return reduce(all_eigmax; init=nothing) do out, cur
        if !(isnothing(out) || isnothing(cur))
            out == cur || throw(ArgumentError("All EigMax must match exactly, got $out and $cur"))
        end
        isnothing(out) ? cur : out
    end
end
_get_eigmax(m::EigMax) = m
_get_eigmax(m::SensitivityAnalysis) = metric(m) isa EigMax ? metric(m) : nothing
_get_eigmax(m::Measure) = nothing

###########################################################################################
# Variable generation for ConnectedGraphInit
    
function _transitionprobability(L::SparseMatrixCSC)
    source_sums = readonlyarray(vec(sum(L, dims=2)))
    source_scaling = inv.(source_sums)
    P = Diagonal(source_scaling) * L

    return P, source_sums
end
# Connectedstochastic
function _substochasticmatrix(rsp::RSP, P::SparseMatrixCSC, C::SparseMatrixCSC)
    @assert LinearAlgebra.checksquare(C) == LinearAlgebra.checksquare(P)
    W = P .* exp.(-theta(rsp) .* C)
    # Any NaNs become zero probabilities
    replace!(W.nzval, NaN => 0.0)
    return W
end

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

# function _check_z(ti::TargetInit{<:RSP})
#     # Check that values in Z are not too small
#     # TODO: does this make sense for single targets
#     if check(ti) && minimum(ti.Z) * minimum(nonzeros(ti.CW)) == 0
#         @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
#     end
# end

#################################################################################3
# Target level variable precalculation

@inline function target_precalculation!(ti::TargetInit{<:RSP}, x::Symbol)::ReadOnlyArray
    st = storage(ti)
    # Either retrieve from storage
    haskey(st, x) && return st[x]
    pre = precalculation(ti)

    # Or compute and store
    st[x] = output = if x === :Z
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
    st = storage(ti)

    # Either retrieve from storage
    haskey(st, x) && return st[x]

    # Or compute and store
    st[x] = output = if x === :Z
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
    elseif x === :IW_factorization
        # For RandomWalk these have to be updated per-target
        # using Woodbury matrices, rather than being defined
        # only at the ConnectedGraph level.
        _woodburysubtochasticmatrixcol(ti)
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
        _landscapematrixcol(ti)
    elseif x === :Q
        _qualitymatrixcol(ti)
    else
        error("Unknown property $x")
    end
    st[x] = output
    return output
end

###########################################################################################
# Variable generation for TargetInit

# Copy a column  from a precalculated full matrix
_copyrow(A, ti) = readonlyarray(workspace(ti) .= view(A, targetconnectedgraphidx(ti), :))
_copycol(A, ti) = readonlyarray(workspace(ti) .= view(A, :, targetconnectedgraphidx(ti)))

function _proximitymatrixcol(ti::TargetInit{<:Union{RSP,RandomWalk}})::RVDe
    pm = proximity_measure(ti)
    dt = distance_transformation(ti)
    distances = get_or_compute_target!(ti, pm)
    proximities = if pm isa DistanceMeasure && !isnothing(dt)
        workspace(ti) .= dt.(distances)
    else
        workspace(ti) .= distances
    end
    _maybe_set_diagonal!(proximities, ti)

    return readonlyarray(proximities)
end

function _fundamentalmatrixcol(ti::TargetInit{<:Union{RSP,RandomWalk}})
    # `Z = (I - W) \ i` where b is column of the a diagonal matrix of 1s
    # Solving `(I - W) * Z = i` for unknown `Z`
    i = _identity_col!(workspace(ti), target(ti))
    i_copy = _identity_col!(workspace(ti), target(ti))
    Z = ldiv!(solver(ti), i, ti.IW_factorization, i_copy)
    return readonlyarray(Z)
end

function _inversefundamentalmatrixcol(ti)
    readonlyarray(_inv!(workspace(ti), ti.Z))
end

function _fundamentalrowmatrixrow(ti::TargetInit{<:Union{RSP,RandomWalk}})
    Zrows = if _issquare(ti)
        ti.Z
    else
        b = _identity_col!(workspace(ti), target(ti))
        Zrows = ldiv!(ti, ti.IW_adj_factorization, b)
        readonlyarray(Zrows)
    end

    return Zrows
end

# TODO: is this the most correct name for Y ?
function _costdistancematrixcol(ti)
    (; CW, Z, IW_factorization) = ti
    # Solve: (I - W) \ (C .* W) * Z ./ Z
    # Manual matmul is *much* faster with sparse/dense.
    # Otherwise this is 99% of the run time.
    RHS = fill!(workspace(ti), 0.0)
    foreachnz(CW) do i, j, n
        RHS[i] += CW.nzval[n] * Z[j] 
    end

    Y = ldiv!(ti, IW_factorization, RHS)

    return readonlyarray(Y)
end

function _landscapematrixcol(ti::TargetInit)
    (; qˢ, K::RVDe, qᵗ::Float64) = ti
    return readonlyarray(workspace(ti) .= qˢ .* K .* qᵗ)
end

function _qualitymatrixcol(ti::TargetInit)
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


# Fill a vector with zeros, and one for the target node
# If this column it was part of a square matrix it would be an identity matrix
function _identity_col!(workspace, target::TargetID)
    fill!(workspace, 0.0)
    workspace[target.node] = 1.0
    return workspace
end


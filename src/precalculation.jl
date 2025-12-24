
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
needs_edgebetweenness_workspace(::Measure, ::MovementMode) = false
num_vec_workspaces(::Measure, ::MovementMode) = 0
num_sp_workspaces(::Measure, ::MovementMode) = 0

# Base sparse workspace counts for sparse_precalculation per movement mode
# RSP: P, W, IW, Aⁱ, CW, CW_t (+ IW_adj for LinearSolver)
# RandomWalk: P, Lⁱ, PC, IW_base (substochastic base for Woodbury)
# LCP: P
num_sp_workspaces_base(::RSP, solver) = solver isa ColumnSolver ? 6 : 7
num_sp_workspaces_base(::RandomWalk, ::Any) = 4
num_sp_workspaces_base(::LCP, ::Any) = 1
num_sp_workspaces_base(::MovementMode, ::Any) = 0

anymeasure(f, measures::NamedTuple, mov::MovementMode) =
    anymeasure(f, values(measures), mov)
anymeasure(f, measures::Tuple{Vararg{Measure}}, mov::MovementMode) =
    any(map(m -> f(m, mov), measures)) 

# TODO: Move these to a precalculation.jl file with the computation precalculations
function sparse_precalculation(problem::ConScapeProblem{<:RSP}, graph::ConnectedGraph, workspaces::WorkspaceCollection)
    _check_inputs(problem, graph)
    A = steplikelihood(graph)
    C = stepcost(graph)
    P, A_rowsums = _transitionprobability(A, sp_workspace(workspaces))
    W = _substochasticmatrix(movement(problem), P, C, sp_workspace(workspaces))
    IW = sp_workspace(workspaces) .= I - W
    F_IW = init(solver(problem), IW)
    Aⁱ = sp_workspace(workspaces)
    Aⁱ.nzval .= inv.(A.nzval)
    if solver(problem) isa ColumnSolver
        IW_adj = IW'
        # Use adjoint factorization of A rather than recalculating for A'
        F_IW_adj = F_IW'
    else # LinearSolver
        # LinearSolve.jl cant handle the adjoint
        # so we duplicate work and allocations
        IW_adj = sp_workspace(workspaces) .= IW'
        F_IW_adj = init(solver(problem), IW_adj)
    end
    CW = sp_workspace(workspaces) .= C .* W
    CW_t = sp_workspace(workspaces) .= transpose(CW)

    θ = theta(problem)
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    return (; P, W, IW, IW_adj, C, CW, CW_t, F_IW, F_IW_adj, A, Aⁱ, A_rowsums, θ, qᵗ, qˢ)
end
function sparse_precalculation(p::ConScapeProblem{<:LCP}, graph::ConnectedGraph, workspaces::WorkspaceCollection)
    _check_inputs(p, graph)
    P, L_rowsums = _transitionprobability(steplikelihood(graph), sp_workspace(workspaces))
    # TODO: use a raster based shortest path algorithm from Geomorphometry.jl
    # disjkstra is especially slow due to allocations,
    # searchsorted for index lookups, and Dict getindex/setindex!.
    cost_weighted_digraph = SimpleWeightedDiGraph(stepcost(graph))
    dsp1 = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, 1)
    parents = dsp1.parents
    path_allocs = Vector{eltype(parents)}[Vector{eltype(parents)}() for _ in eachindex(parents)]

    # For completeness we also move these to precalculations
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    (; P, L_rowsums, cost_weighted_digraph, path_allocs, qᵗ, qˢ)
end
function sparse_precalculation(problem::ConScapeProblem{<:RandomWalk}, graph::ConnectedGraph, workspaces::WorkspaceCollection)
    _check_inputs(problem, graph)
    L = steplikelihood(graph)
    P, L_rowsums = _transitionprobability(L, sp_workspace(workspaces))
    Lⁱ = sp_workspace(workspaces)
    Lⁱ.nzval .= inv.(L.nzval)
    PC = sp_workspace(workspaces) .= P .* stepcost(graph)
    PC_rowsums = sum(PC; dims=2)

    # RandomWalk Woodbury base matrix strategy:
    #
    # For RandomWalk, we need (I - W_t)^{-1} for each target t, where W_t is P with row t zeroed.
    #
    # PROBLEM: Using I-P as the Woodbury base matrix fails because P is stochastic (rows sum to 1),
    # making I-P singular (eigenvalue 0). The LU factorization has near-zero pivots (~1e-16),
    # causing catastrophic numerical errors in the Woodbury formula for some targets.
    #
    # SOLUTION: Use I-W₁ (substochastic, with first target row zeroed) as the base matrix.
    # W₁ has spectral radius < 1, so I-W₁ is well-conditioned and invertible.
    # For other targets, we use a rank-2 Woodbury update: IW_t = IW₁ + U*V where:
    #   - U = [e₁ | e_t]  (restore row 1, zero row t)
    #   - V = [P[1,:]; -P[t,:]]
    #
    # This gives numerically stable solves for all targets.

    # Get the first target node to use as the base for the substochastic matrix
    base_target_node = first(targetids(graph)).node

    # Create W₁ = P with base target row zeroed (substochastic)
    IW_base = sp_workspace(workspaces) .= I - P
    # Zero the base target row in the sparse matrix by setting its values to the identity row
    # IW_base[base_target_node, :] should be [0, ..., 1, ..., 0] (identity row)
    for j in 1:size(IW_base, 2)
        if j == base_target_node
            IW_base[base_target_node, j] = 1.0
        else
            IW_base[base_target_node, j] = 0.0
        end
    end
    F_IW_base = init(solver(problem), IW_base)

    # For completeness we also move these to precalculations
    qᵗ = targetquality(graph)
    qˢ = sourcequality(graph)

    return (; Lⁱ, L_rowsums, P, PC, PC_rowsums, IW_base, F_IW_base, base_target_node, qᵗ, qˢ)
end
function sparse_precalculation(::ConScapeProblem{<:Euclidean}, ::ConnectedGraph, ::WorkspaceCollection)
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
            mat_workspace(cgi) 
        end
        Zrows_full = if anymeasure(needs_full_fundamentalrowmatrix, mes, mov)
            _issquare(cgi) ? Z_full : reshape(mat_workspace(cgi), reverse(size(Z_full)))
        end
        Y_full = if anymeasure(needs_full_costdistancematrix, mes, mov)
            mat_workspace(cgi) 
        end

        # Precalculate required matrices column by column
        for target in targetids(cgi)
            ti = TargetInit(cgi, target)
            t = target.connectedgraphidx

            if !isnothing(Z_full)
                # Zrows (square) and Y need Z so we store it
                Z = _fundamentalmatrixcol(ti)
                storage(ti)[:Z] = Z
                @views Z_full[:, t] .= Z
            end
            if !isnothing(Zrows_full) && !_issquare(cgi)
                @views Zrows_full[t, :] .= _fundamentalrowmatrixrow(ti)
            end
            if !isnothing(Y_full)
                @views Y_full[:, t] .= _costdistancematrixcol(ti)
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
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), 
                workspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
            )
            em = _get_eigmax(mes)::EigMax
            eigmax = _compute_eigmax(em, cgi_part_precalc)
            precalculation = (; precalculation..., eigmax)
        end

        # EigMax sensitivity precursors
        if anymeasure(needs_sum_sensitivity_precursors, mes, mov)
            cgi_part_precalc = ConnectedGraphInit(
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), 
                workspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
            )
            sum_sensitivity_precursors = _compute_sum_sensitivity_precursors(proximity_measure(cgi), cgi_part_precalc)
            precalculation = (; precalculation..., sum_sensitivity_precursors)
        end

        # Summation sensitivity precursors
        if anymeasure(needs_eigmax_sensitivity_precursors, mes, mov)
            cgi_part_precalc = ConnectedGraphInit(
                problem(cgi), gridgraph(cgi), connectedgraph(cgi), measures_outputs(cgi), 
                workspaces(cgi), storage(cgi), precalculation, connectedgraphid(cgi),
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
    
function _transitionprobability(L::SparseMatrixCSC, P::SparseMatrixCSC)
    L_rowsums = readonlyarray(vec(sum(L, dims=2)))
    # P = Diagonal(inv.(L_rowsums)) * L
    foreachnz(L) do i, j, n
        P.nzval[n] = L.nzval[n] / L_rowsums[i]
    end
    return P, L_rowsums
end

function _substochasticmatrix(rsp::RSP, P::SparseMatrixCSC, C::SparseMatrixCSC, W::SparseMatrixCSC)
    @assert LinearAlgebra.checksquare(C) == LinearAlgebra.checksquare(P)
    # W = P .* exp.(-θ .* C)
    # Note: At very high theta (>5-10), exp(-θ*C) underflows and the RSP model degenerates
    # (random walk barely moves). Use LCP for deterministic shortest paths instead.
    W.nzval .= P.nzval .* exp.(.-theta(rsp) .* C.nzval)
    replace!(W.nzval, NaN => 0.0)
    return W
end

# Computes the stationary distribution of a random walk following the transition probability matrix
function _stationary_distribution(solver::Solver, P::SparseMatrixCSC)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    F_PI = init(solver, PI)
    PI[1, :] .= 1
    v = zeros(n)
    v1 = zeros(n)
    v[1] = v1[1] = 1
    return ldiv!(solver, v, F_PI, v1)
end

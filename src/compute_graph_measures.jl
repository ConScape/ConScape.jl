#=
`compute` and `compute_target` for GraphMeasures.  eventually the whole package could work somthing like this
with computations all at the single target level and 
aggregation processes controlled with `returntrait`
=#

"""
    compute(::MovementMode, gm::GraphMeasure, g::GridPrecalculations[, target::Int)

Computes results of a single target pixel for a graph measure
and movement mode. 

`compute` is called inside `compute` or in the loop in `solve!` for 
VectorSolver/LinearSolver. 
"""
function compute end

const Targets = @NamedTuple{spatialid::CartesianIndex{2},targetid::Int,targetnode::Int}

# We specialise on returntrait
function compute(graph_measures::NamedTuple, gp::GridPrecalculation)
    # Precalculate the sparse grid for this movement mode and all graph measures
    grid_precalculations = precalculate(movement, graph_measures, grid)
    # Generate outputs for each graph measure
    outputs = map(gm -> allocate_output(gm, grid), graph_measures)
    # Loop over targets
    for (i, t) in enumerate(g.targetids)
        # Specify target indices
        target = Target(g.id_to_grid_coordinate_list[i], t, i)
        # Precalculate for this target and graph measures
        target_precalculations = precalculate(grid_precalculations, gms, target)
        # Compute for this target and graph measures
        foreach(graph_measures, outputs) do gm, output
            # TODO: these are probably wrong
            # Compute the graph measure for this target
            v = compute(gm, target_precalculations)
            # Write values to output depending on returntrait
            _update!(output, gm, v, target)
        end
    end

    return outputs
end

# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure
_update!(output, gm, v, target) = _update!(output, returntrait(gm), v, target) 
_update!(output, ::AssignDenseSpatial, v::Number, target) = output[target.spatialid] = v
_update!(output, ::SumDenseSpatial, v::AbstractMatrix, target) = output .+= v
_update!(output, ::AssignSparse, v::AbstactVector, target) = output[target.targetnode, :] = v
function _update!(output::Tuple, ::ReturnsEigMax, v::Tuple, target)
    output[1][target.targetnode] = v
    output[3][target.targetnode] = v
end


# LeastCost
function compute(::LeastCost, gm::Betweenness, g::GridPrecalculations, t::Targets)
    # Calculate distances
    (; cost_weighted_digraph) = g.cost_weighted_digraph # simpleweighteddigraph(g.costmatrix)
    shorted_paths = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, t.targetid)
    shortest_paths_en = Graphs.enumerate_paths(shorted_paths)
    k = ones(size(g.costmatrix, 1)) # TODO use a workspace
    # And maybe transform them
    # TODO untangle this logic so apply_weight handles everythign
    if !isnothing(distance_transformation(cm))
        k .= distance_transformation(cm).(shorted_paths.dists)
    end
    apply_weight!(k, gm, g, t.targetid)

    # TODO what does all this do...
    shortest_paths_en[t.targetid] = [t.targetid]
    tgts = [repeat([i], length(dijk[i])) for i in (1:length(shorted_paths))]
    tgts = reduce(vcat, tgts)
    final_paths = reduce(vcat, final_paths)

    btw = sparse(final_paths, tgts, repeat([1], length(tgts)))

    return btw * k
end

# RandomWalk 
function compute(::EdgeBetweenness{Weighting}, tp::RandomWalkTargetPrecalculations) where Weighting
    nodebet = compute(Betweenness(Weighting()), tp)
    return nodebet * pref[t.spatialid]
end
function compute(::RandomWalk, ::Betweenness{QualityWeighted}, g::GridPrecalculations, t::Targets)
    Z, H, qˢ, qᵗ, p, workspace = g.fundamental, g.hitting_time, g.source_quality, g.target_quality, g.stationary_distribution, g.workspaces[1]
    return qˢ' * _betweenness!(workspace, Z, H, p, t) * qᵗ
end
function compute(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}, g::GridPrecalculations, t::Targets)
    Z, H, K, p, workspace = g.probability, g.cost, g.fundamental, g.hitting_time, g.quality_weighted_proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end
function compute(::RandomWalk, ::Betweenness{ProximityWeighted}, g::GridPrecalculations, t::Targets)
    Z, H, K, p, workspace = g.fundamental, g.hitting_time, g.proximity, g.stationary_distribution, g.workspaces[1]
    return sum(_betweenness!(workspace, Z, H, p, t) .* K)
end

# TODO this only works for simmetrical Z
_betweenness!(workspace, Z, H, p, t::Targets) =
    workspace .= view(Z, :, 1) .- view(Z, :, t)' .+ H .* p[t]

# This is an idea for specifying precomputed arrays needed for a given graph measure
# It would be nice to have this close top the algorithme.
# This function could also be defined with a @needs macro on the 
# `compute` function to remove the name duplication
needs(::RandomWalk, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :source_quality, :target_quality, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{QualityAndProximityWeighted}) = 
    (:fundamental, :hitting_time, :quality_weighted_proximity, :stationary_distribution, :workspaces => 1)
needs(::RandomWalkd, ::Betweenness{ProximityWeighted}) = 
    (:fundamental, :hitting_time, :proximity, :stationary_distribution, :workspaces => 1)

apply_weight!(k, ::Unweigthed, g::TargetPrecalculations) = k
apply_weight!(k, ::QualityAndProximityWeighted, g::TargetPrecalculations) =
    k .*= g.qˢ .* g.[targets.targetnode]

# RandomShortestPath

# Betweenness: TODO move code here from RSP and rewrite per-target
function compute(::Betweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    (; Z, qˢ, Zⁱ, IWadj, IWadj_factorization) = tp
    workspace1, workspace2 = tp.workspaces
    qᵗ = g.target_qualities[target.targetnode]

    qˢZⁱqᵗ = workspace1 .= qˢ .* Zⁱ .* qᵗ
    # TODO: explain why this is needed
    qˢZⁱqᵗ[target.targetnode, 1] -= sum(qˢ) * qᵗ * Zⁱ[target.targetnode, 1]

    ZqˢZⁱqᵗZt = ldiv!(tp.solver, Aadj_init, qˢZⁱqᵗ; B_copy=copy!(workspace2, qˢZⁱqᵗ))
    ZqˢZⁱqᵗZt .*= Z

    return ZqˢZⁱqᵗZt
end
function compute(::Betweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    (; Z, M, qˢ, Zⁱ, IWadj, IWadj_factorization) = tp
    workspace1, workspace2 = tp.workspaces
    qᵗ = g.target_qualities[target.targetnode]
    workspace1, workspace2 = tp.workspaces

    # Divide the  by Z
    MZⁱ = workspace2 .= M .* Zⁱ

    # Find the scaling factor:
    # If any of the values of KZⁱ is above one then there is a risk of overflow,
    # so we scale the matrix and apply a scale factor
    λ = max(1.0, maximum(MZⁱ))
    # TODO: what is this for
    k = sum(MZⁱ) / λ
    MZⁱ[target.targetnode, 1] -= k * Zⁱ[target.targetnode, 1]
    # Normalise before the solve
    MZⁱ ./= λ
    # Solve: ZKZⁱt = (I - W)' \ KZⁱ
    ZMZⁱt = ldiv!(solver, IWadj_init, MZⁱ; B_copy=copy!(workspace1, MZⁱ))
    # Rescale after the solve
    ZMZⁱt .*= λ .* Z

    return sum(ZMZⁱt)
end
function compute(::EdgeBetweenness{QualityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    (; Z, Zⁱ, qˢ, IWadj_factorization) = tp
    workspace1, workspace2, workspace3 = tp.workspaces

    qᵗ = g.target_qualities[target.targetnode]
    diagZⁱ = Zⁱ[t.targetnode, 1]
    
    B = workspace1 .= B_sparse
    Zrows = ldiv!(solver, IWadj_factorization, B; B_copy=copy!(workspace2, B))'
    Zrows .*= sum(qˢ) * qᵗ .* diagZⁱ

    qˢZⁱqᵗ = workspace2 .= qˢ .* Zⁱ .* qᵗ
    # QZⁱᵀZ = qˢZⁱqᵗ' / A
    QZⁱᵀZ = ldiv!(solver, Aadj_init, qˢZⁱqᵗ; B_copy=copy!(workspace3, qˢZⁱqᵗ))'
    RHS = workspace3 .= QZⁱᵀZ .- Zrows

    return _edge_betweenness(W, Z, RHS, target)
end
function compute(::EdgeBetweenness{QualityAndProximityWeighted}, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    M = tp.landscape_matrix
    workspace1, workspace2 = tp.workspaces
    permuted_workspace1 = tp.permuted_workspaces[1]
    k̂ = sum(M̂)
    K̂ .*= Zⁱ

    # K̂ᵀZ =  K̂' / A # is equivalent to the below
    K̂ᵀZ = ldiv!(solver(tp), IWadj_init, K̂; B_copy=copy!(workspace2, K̂))'

    k̂diagZⁱ = k̂ * Zⁱ[target.targetnode, 1]

    B = workspace1 .= B_sparse
    Zrows = ldiv!(solver, Aadj_init, B; B_copy=copy!(workspace2, B))
    K̂ᵀZ_minus_diag = permuted_workspace1 .= K̂ᵀZ .- k̂diagZⁱ .* Zrows'

    return _edge_betweenness(W, Z, K̂ᵀZ_minus_diag, target)
end

function _edge_betweenness(W, Z, X, target)
    t = target.targetnode
    edge_betweennesses = spzeros(size(W, 1))
    for i in axes(W, 1)
        x = W[i, t]
        if x > 0 
            edge_betweennesses[i] = x * Z[j, t] * X[1, i]
        end
    end
end

# ConnectedHabitat 
compute(::ConnectedHabitat, tp::TargetPrecalculations) = tp.landscape_matrix

function compute(gm::EigMax, tp::RandomisedShortestPathTargetPrecalculations)
    g = grid(tp)
    M = tp.landscape_matrix
    workspace1 = tp.workspaces
    vˡ = zeros(n)
    vʳ = fill(NaN, n)

    vʳ .= NaN
    # Square submatrix defined by extracting the rows corresponding to landmarks
    M₀₀ = view(workspace1, 1:1, :)
    M₀₀ .= view(M, t:t, :) # TODO: does this need to be a matrix?

    # Size of the full problem
    n = size(affinities(tp), 1)

    # Use an Arnoldi based eigensolver to compute the largest (absolute) eigenvalue and right vector (of submatrix)
    Fps = partialschur(M₀₀; nev=1, tol=gm.tol)
    λ₀, vʳ₀ = partialeigen(Fps[1])

    # Construct full right vector
    vʳ[t] = vʳ₀
    for i in 1:n
        i == t.targetnode && continue
        vʳ[i] = view(M, i, :) * vʳ₀ / λ₀[1]
    end

    # Compute left vector (of submatrix) by shift-invert
    # TODO rewrite for single target
    Flu = lu(M₀₀ - λ₀[1] * I)
    vˡ₀ = ldiv!(Flu', rand(1))
    rmul!(vˡ₀, inv(vˡ₀[1]))
    vˡ[t.targetnode] = vˡ₀

    return (vˡ, λ₀=λ₀[1], vʳ)
end        

function compute(gm::Sensitivity, tp::RandomisedShortestPathTargetPrecalculations)
    if gm.landscape_measure <: LandscapeEigen
        v, λ, w = compute(EigMax(), tp)
        vTw = (v') * w
        if !(gm.context <: Quality)
            qˢ = qˢ .* v 
            qᵗ = qᵗ .* w
        end
    end

    target_sensitivity = if gm.context <: Union{Affinity,Cost,CostAndAffinitySensitivityContext} 
        S_e_aff, S_e_cost = if cm <: ExpectedCost
            diff_K_D = _diff_KD(distance_transformation)
            EC_sensitivity(g.affinities, g.costmatrix, m.θ, tp.W, tp.Z, diff_K_D, qˢ, qᵗ, [t])[1]
        else
            PM_sensitivity(g.affinities, nothing, m.θ, tp.W, tp.Z, tp.Z, qˢ, qᵗ, [t])[1]
        end

        if unitless 
            _scale!(S_e_aff, gm.context, g)
            _scale!(S_e_cost, gm.context, g)
        end

        if gm.context <: Affinity
            sum(S_e_aff)
        elseif gm.context <: Cost
            sum(S_e_cost)
        elseif gm.context <: AffinityAndCost
            diff_C_A = ConScape.mapnz(diff_C_A_fun, g.affinities)
            S_e_total = S_e_aff .+ S_e_cost .* diff_C_A
            sum(S_e_total)
        elseif gm.context <: CostAndAffinity
            diff_A_C = ConScape.mapnz(diff_A_C_fun, g.affinities)
            S_e_total = S_e_aff .* diff_A_C .+ S_e_cost
            sum(S_e_total)
        end
    elseif gm.context <: Qualities
        K = tp.proximities
        if landscape_measure <: LandscapeEigen
            K = v .* K .* (w')
        end
        K = K .+= transpose(K)
        target_sensitivity = K * g.target_qualities[I]
        if unitless
            target_sensitivity *= g.source_qualities[I]
        end
        target_sensitivity
    end

    if gm.landscape_measure <: LandscapeEigen
        target_sensitivity /= vTw
    end

    output[I] = target_sensitivity * isnan(g.source_qualities[I]) ? NaN : 1
    return output
end

_diff_CA_fun(::MinusLog) = x -> -inv(x)
_diff_AC_fun(::MinusLog) = x -> -(x)
_diff_CA_fun(::Inv) = x -> -inv(x^2) # TODO fix
_diff_AC_fun(::Inv) = x -> -inv(x^2)

_diff_KD(x::ExpMinusAlpha) = -K .* x.α
_diff_KD(::Inv) = -K .^ 2

# TODO a more specific name
_scale!(A, ::Union{Affinity,AffinityAndCost}, g) = A .*= g.affinities
_scale!(A, ::Union{Cost,CostAndAffinity}, g) = A .*= g.costmatrix


"""
    EC_sensitivity(A::SparseMatrixCSC,
        C::SparseMatrixCSC,
        θ::Real,
        W::SparseMatrixCSC,
        Z::AbstractMatrix,
        K::AbstractMatrix,  # diff_K_D (but can be any weighting matrix)
        qˢ::AbstractVector,
        qᵗ::AbstractVector,
        lmarks::AbstractVector)

Expected cost sensitivity
"""
function EC_sensitivity(A::SparseMatrixCSC,
    C::SparseMatrixCSC,
    θ::Real,
    W::SparseMatrixCSC,
    Z::AbstractMatrix,
    K::AbstractMatrix,  # diff_K_D (but can be any weighting matrix)
    qˢ::AbstractVector,
    qᵗ::AbstractVector,
    lmarks::AbstractVector)

    Zⁱ = inv.(Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Z)) # To prevent Inf*0 later...

    K̂ = qˢ .* K .* qᵗ'
    k̂ = vec(sum(K̂, dims=1))
    K̂ .*= Zⁱ # k̂ᵢⱼ = kᵢⱼ/zᵢⱼ

    CW = C .* W

    Y = (ConScape.I - W) \ (CW * Z)

    C̄ᵣ = Y .* Zⁱ # Expected costs of REGULAR paths

    K̂ᵀZ = K̂' / (ConScape.I - W)

    k̂diagZⁱ = k̂ .* [Zⁱ[lmarks[t], t] for t in 1:length(lmarks)]

    if size(Z, 2) < size(Z, 1)
        I_L = Matrix(sparse(lmarks, 1:length(lmarks), 1.0, size(W, 1), length(lmarks)))
        Zrows = I_L' / (ConScape.I - W)
    else
        Zrows = Z
    end

    X3 = k̂diagZⁱ .* Zrows

    k̂diagC̄Zⁱ = k̂diagZⁱ .* [C̄ᵣ[lmarks[t], t] for t in 1:length(lmarks)]
    X5 = ((K̂ .* C̄ᵣ)' - (K̂ᵀZ * CW) + (X3 * CW)) / (ConScape.I - W) - k̂diagC̄Zⁱ .* Zrows # "X1- X2 - X4"

    X3 .= K̂ᵀZ .- X3

    kΣ = copy(W) # k-weighted negative covariance matrix
    kB = copy(W) # k-weighted edge betweenness matrix

    for i in axes(W, 1)
        for j in findall(W[i, :] .> 0)
            kB[i, j] *= (Z[j, :]'*X3[:, i])[1]
            kΣ[i, j] *= (Z[j, :]'*X5[:, i])[1] - (Y[j, :]'*X3[:, i])[1] - C[i, j] * kB[i, j] / W[i, j]
        end
    end

    kΣ_node = sum(kΣ, dims=2)
    Ae = sum(A, dims=2)

    S_cost = kB + θ * kΣ#/θ

    Idx = W .> 0
    Aⁱ = ConScape.mapnz(x -> inv(x), A)
    S_aff = (kΣ_node ./ Ae) .* Idx - kΣ .* Aⁱ

    return S_aff, S_cost
end


"""
    PM_sensitivity(A::SparseMatrixCSC,
        C::Union{Nothing, SparseMatrixCSC},
        θ::Real,
        W::SparseMatrixCSC,
        Z::AbstractMatrix,
        K::AbstractMatrix,  #K = copy(Z)
        qˢ::AbstractVector,
        qᵗ::AbstractVector,
        lmarks::AbstractVector)

Power mean sensitivity, which includes survival.
"""
function PM_sensitivity(A::SparseMatrixCSC,
    C::Union{Nothing,SparseMatrixCSC},
    θ::Real,
    W::SparseMatrixCSC,
    Z::AbstractMatrix,
    K::AbstractMatrix,  #K = copy(Z)
    qˢ::AbstractVector,
    qᵗ::AbstractVector,
    lmarks::AbstractVector)

    K ./= [Z[lmarks[i], i] for i in 1:length(lmarks)]'
    K .^= θ # \mathcal{Z}^θ

    rowsums = sum(A, dims=2)

    bet_edge_k = ConScape.RSP_edge_betweenness_kweighted(W, Z, qˢ, qᵗ, K, lmarks)
    S_e_cost = -bet_edge_k

    bet_node_k = ConScape.RSP_betweenness_kweighted(W, Z, qˢ, qᵗ, K, lmarks)

    Idx = A .> 0
    Aⁱ = ConScape.mapnz(inv, A)
    S_e_aff = (bet_edge_k .* Aⁱ .- (bet_node_k ./ rowsums) .* Idx) .* θ

    return S_e_aff, S_e_cost

end


"""
    _Imn(n::Integer, landmarks::AbstractVector)

Helper function to compute a (column subset of a) dense identity matrix where the subset corresponds
to the landsmarks
"""
function _Imn(n::Integer, landmarks::AbstractVector)
    Imn = zeros(n, length(landmarks))
    for (j, i) in enumerate(landmarks)
        Imn[i, j] = 1
    end
    return Imn
end
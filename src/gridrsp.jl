"""
    betweenness_qweighted(grsp::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all nodes weighted by source and target qualities.
"""
function betweenness_qweighted(grsp::Union{GridRSP,NamedTuple};
    output=_init_output(grsp.g),
    kw...
)
    g = grsp.g
    betvec = RSP_betweenness_qweighted(grsp.W, grsp.Z, g.qs, g.qt, g.targetnodes; kw...)
    _update_output!(output, g, betvec)
    return _maybe_raster(output, g)
end

function _update_output!(output, g, betvec)
    for (I, v) in zip(g.id_to_grid_coordinate_list, betvec)
        x = output[I]
        output[I] = isnan(x) ? v : x + v
    end
end

"""
    edge_betweenness_qweighted(grsp::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all edges weighted by source and target qualities. Returns a
sparse matrix where element (i,j) is the betweenness of edge (i,j).
"""
function edge_betweenness_qweighted(grsp::Union{GridRSP,NamedTuple}; kw...)
    g = grsp.g
    return RSP_edge_betweenness_qweighted(grsp.W, grsp.Z, g.qs, g.qt, g.targetnodes; kw...)
end

"""
    betweenness_kweighted(grsp::GridRSP;
        connectivity_function=expected_cost,
        distance_transformation=inv(grsp.g.costfunction),
        diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

Compute RSP betweenness of all nodes weighted with proximities computed with 
respect to the distance/proximity measure defined by `connectivity_function`. 
Optionally, an inverse cost function can be passed. The function will be applied 
elementwise to the matrix of distances to convert it to a matrix of proximities. 
If no inverse cost function is passed the the inverse of the cost function is 
used for the conversion of distances.

The optional `diagvalue` element specifies which value to use for the diagonal 
of the matrix of proximities, i.e. after applying the inverse cost function to the 
matrix of distances. When nothing is specified, the diagonal elements won't be adjusted.
"""
function betweenness_kweighted(grsp::Union{GridRSP,NamedTuple};
    output=_init_output(grsp.g),
    proximities=nothing,
    kw...
)
    g = grsp.g
    if isnothing(proximities)
        proximities = _computeproximities(grsp; kw...)
    end

    betvec = RSP_betweenness_kweighted(grsp.W, grsp.Z, g.qs, g.qt, proximities, g.targetnodes; kw...)
    _update_output!(output, g, betvec)
    return _maybe_raster(output, g)
end


"""
    edge_betweenness_kweighted(grsp::GridRSP; [distance_transformation=inv(grsp.g.costfunction), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

    Compute RSP betweenness of all edges weighted by qualities of source s and target t and the proximity between s and t. Returns a sparse matrix where element (i,j) is the betweenness of edge (i,j).

    of proximities, i.e. after applying the inverse cost function to the matrix of expected costs.
    When nothing is specified, the diagonal elements won't be adjusted.
"""
function edge_betweenness_kweighted(grsp::Union{GridRSP,NamedTuple};
    proximities=nothing,
    distance_transformation=nothing,
    diagvalue=nothing,
    kw...
)
    if isnothing(distance_transformation)
        distance_transformation = inv(grsp.g.costfunction)
    end
    # TODO why does this only use `expected_cost`?
    g = grsp.g
    # S = map(distance_transformation, expected_cost(grsp))
    proximities = map(distance_transformation, expected_cost(grsp))
    maybe_set_diagonal!(proximities, diagvalue, g.targetnodes)

    betmatrix = RSP_edge_betweenness_kweighted(grsp.W, grsp.Z, g.qs, g.qt, proximities, g.targetnodes; kw...)
    return betmatrix
end

"""
    expected_cost(grsp::GridRSP)::Matrix{Float64}

Compute RSP expected costs from all nodes.
"""
expected_cost(grsp::Union{GridRSP,NamedTuple}; kw...) =
    RSP_expected_cost(grsp.W, grsp.g.costmatrix, grsp.Z, grsp.g.targetnodes; kw...)

free_energy_distance(grsp::Union{GridRSP,NamedTuple}; kw...) =
    RSP_free_energy_distance(grsp.Z, grsp.θ, grsp.g.targetnodes; kw...)

survival_probability(grsp::Union{GridRSP,NamedTuple}; kw...) =
    RSP_survival_probability(grsp.Z, grsp.θ, grsp.g.targetnodes; kw...)

power_mean_proximity(grsp::Union{GridRSP,NamedTuple}; kw...) =
    RSP_power_mean_proximity(grsp.Z, grsp.θ, grsp.g.targetnodes; kw...)

least_cost_distance(grsp::Union{GridRSP,NamedTuple}; kw...) = least_cost_distance(grsp.g; kw...)

"""
    mean_kl_divergence(grsp::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the free 
energy distances and the RSP expected costs for `grsp::GridRSP`.
"""
function mean_kl_divergence(grsp::Union{GridRSP,NamedTuple};
    free_energy_distances=nothing,
    expected_costs=nothing,
    kw...
)
    g = grsp.g
    free_energy_distances = if isnothing(free_energy_distances)
        RSP_free_energy_distance(grsp.Z, grsp.θ, g.targetnodes; kw...)
    else
        free_energy_distances
    end
    expected_costs = if isnothing(expected_costs)
        ConScape.expected_cost(grsp; kw...)
    else
        expected_costs
    end
    return mean_kl_divergence(grsp::Union{GridRSP,NamedTuple}, free_energy_distances, expected_costs; kw...)
end
function mean_kl_divergence(grsp::Union{GridRSP,NamedTuple}, free_energy_distances, expected_costs;
    workspaces=(similar(grsp.Z),), kw...
)
    g = grsp.g
    fed_exp = workspaces[1] .= free_energy_distances .- expected_costs
    return g.qs' * fed_exp * g.qt * grsp.θ
end


"""
    mean_lc_kl_divergence(grsp::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the least-cost path and the random path
distribution for `grsp::GridRSP`, weighted by the qualities of the source and target node.
"""
function mean_lc_kl_divergence(grsp::Union{GridRSP,NamedTuple};
    workspaces=[similar(grsp.Z)],
    kw...
)
    workspace1 = workspaces[1]
    g = grsp.g
    C = g.costmatrix
    cost_weighted_digraph = SimpleWeightedDiGraph(C)
    n = size(C, 1)
    from = Array{Int}(undef, n)
    kl_div = Array{Float64}(undef, n)
    # Previously
    # div = hcat([least_cost_kl_divergence(C, grsp.Pref, i; cost_weighted_digraph, from, kl_div, kw...) for i in g.targetnodes]...)
    div = workspace1
    for i in g.targetnodes
        div[i, :] .= least_cost_kl_divergence(C, grsp.Pref, i; cost_weighted_digraph, from, kl_div, kw...)
    end
    return g.qs' * div * g.qt
end

function least_cost_kl_divergence(C::SparseMatrixCSC, Pref::SparseMatrixCSC, targetnode::Integer;
    cost_weighted_digraph=SimpleWeightedDiGraph(C),
    n=size(C, 1),
    from=Array{Int}(undef, n),
    kl_div=Array{Float64}(undef, n),
    kw...
)
    from .= 1:n
    fill!(kl_div, 0)

    if !(1 <= targetnode <= n)
        throw(ArgumentError("target node not found"))
    end

    dsp = dijkstra_shortest_paths(cost_weighted_digraph, targetnode)
    parents = dsp.parents
    parents[targetnode] = targetnode
    to = copy(parents)

    while true
        notdone = false

        for i in 1:n
            fromᵢ = from[i]
            toᵢ = to[i]
            notdone |= fromᵢ != toᵢ
            if fromᵢ == toᵢ
                continue
            end
            v = Pref[fromᵢ, toᵢ]
            kl_div[i] += -log(v)
            from[i] = parents[toᵢ]
        end
        if !notdone
            break
        end

        # Pointer swap
        tmp = from
        from = to

        to = tmp
    end

    return kl_div
end
"""
    least_cost_kl_divergence(grsp::GridRSP, target::Tuple{Int,Int})

Compute the least cost Kullback-Leibler divergence from each
cell in the g in `h` to the `target` cell.
"""
function least_cost_kl_divergence(grsp::Union{GridRSP,NamedTuple}, target::Tuple{Int,Int}; kw...)
    g = grsp.g
    targetnode = findfirst(isequal(CartesianIndex(target)), g.id_to_grid_coordinate_list)
    if targetnode === nothing
        throw(ArgumentError("target cell not found"))
    end

    div = least_cost_kl_divergence(g.costmatrix, grsp.Pref, targetnode; kw...)

    return reshape(div, g.nrows, g.ncols)
end

"""
    connected_habitat(grsp::Union{Grid,GridRSP};
        connectivity_function=expected_cost,
        distance_transformation=nothing,
        diagvalue=nothing,
        θ::Union{Nothing,Real}=nothing,
        approx::Bool=false)::Matrix{Float64}

Compute RSP connected_habitat of all nodes. An inverse
cost function must be passed for a `Grid` argument but is optional for `GridRSP`.
The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the proximities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `distance_transformation`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`.

For `Grid` objects, the inverse temperature parameter `θ` must be passed when the `connectivity_function`
requires it such as `expected_cost`. Also for `Grid` objects, the `approx` Boolean
argument can be set to `true` to switch to a cheaper approximate solution of the
`connectivity_function`. The default value is `false`.
"""
function connected_habitat(
    grsp::Grid;
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing,
    θ::Union{Nothing,Real}=nothing,
    approx::Bool=false
)
    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        throw(ArgumentError("distance_transformation function is required when passing a Grid together with a Distance function"))
    end

    if θ === nothing && connectivity_function !== least_cost_distance
        throw(ArgumentError("θ must be a positive real number when passing a Grid"))
    end
    proximities = connectivity_function(grsp; θ=θ, approx=approx)
    if connectivity_function <: DistanceFunction
        map!(distance_transformation, proximities, proximities)
    end

    return connected_habitat(grsp, proximities; diagvalue)
end

function connected_habitat(grsp::GridRSP; proximities=nothing, kw...)
    if isnothing(proximities)
        proximities = _computeproximities(grsp; kw...)
    end
    return connected_habitat(grsp, proximities; kw...)
end
connected_habitat(grsp::GridRSP, S::Matrix; kw...) =
    connected_habitat(grsp.g, S; kw...)
function connected_habitat(g::Grid, S::Matrix;
    diagvalue::Union{Nothing,Real}=nothing,
    output=_init_output(g),
    kw...
)
    maybe_set_diagonal!(S, diagvalue, g.targetnodes)

    funvec = connected_habitat(g.qs, g.qt, S; kw...)

    for (I, x) in zip(g.id_to_grid_coordinate_list, funvec)
        output[I] = x
    end

    return _maybe_raster(output, g)
end
function connected_habitat(grsp::GridRSP, cell::CartesianIndex{2};
    distance_transformation=nothing,
    diagvalue=nothing,
    avalue=floatmin(), # smallest non-zero value
    qˢvalue=0.0,
    qᵗvalue=0.0,
    kw...)

    g = grsp.g

    if avalue <= 0.0
        throw("Affinity value has to be positive. Otherwise the graph will become disconnected.")
    end

    # Compute (linear) node indices from (cartesian) grid indices
    node = findfirst(isequal(cell), g.id_to_grid_coordinate_list)

    # Check that cell is in targetidx
    if cell ∉ g.targetidx
        throw(ArgumentError("Computing adjusted connected_habitat is only supported for target cells"))
    end

    affinities = copy(g.affinities)
    affinities[:, node] .= ifelse.(iszero.(affinities[:, node]), 0, avalue)
    affinities[node, :] .= ifelse.(iszero.(affinities[node, :]), 0, avalue)

    newsource_qualities = copy(g.source_qualities)
    newsource_qualities[cell] = qˢvalue
    newtarget_qualities = copy(g.target_qualities)
    newtarget_qualities[cell] = qᵗvalue

    newtargetidx, newtargetnodes = _targetidx_and_nodes(newtarget_qualities, g.id_to_grid_coordinate_list)
    newqs = [newsource_qualities[i] for i in g.id_to_grid_coordinate_list]
    newqt = [newtarget_qualities[i] for i in g.id_to_grid_coordinate_list ∩ newtargetidx]

    newg = Grid(g.nrows,
        g.ncols,
        affinities,
        g.costfunction,
        g.costfunction === nothing ? g.costmatrix : mapnz(g.costfunction, affinities),
        g.id_to_grid_coordinate_list,
        newsource_qualities,
        newtarget_qualities,
        newtargetidx,
        newtargetnodes,
        newqs,
        newqt,
        dims(g))

    newh = GridRSP(newg; θ=grsp.θ)

    return connected_habitat(newh; diagvalue, distance_transformation)
end

"""
    eigmax(grsp::GridRSP;
        connectivity_function=expected_cost,
        distance_transformation=nothing,
        diagvalue=nothing,
        tol=1e-14)

Compute the largest eigenvalue triple (left vector, value, and right vector) of the 
quality scaled proximities with respect to the distance/proximity measure defined by 
`connectivity_function`. 

If `connectivity_function` is a distance measure then the distances are transformed
to proximities by `distance_transformation` which defaults to the inverse of the `costfunction`
in the underlying `Grid` (if defined). Optionally, the diagonal values of the proximity matrix may 
be set to `diagvalue`. The `tol` argument specifies the convergence tolerance in the Arnoldi based eigensolver.
"""
function LinearAlgebra.eigmax(grsp::Union{GridRSP,NamedTuple};
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing,
    workspaces=[similar(grsp.Z), similar(grsp.Z), similar(grsp.Z)],
    tol=1e-14,
    expected_costs=nothing,
    free_energy_distances=nothing,
    kw...
)
    g = grsp.g
    workspace1, workspace2, workspace3 = workspaces

    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        if g.costfunction === nothing
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function."))
        else
            distance_transformation = inv(g.costfunction)
        end
    end

    proximities = if connectivity_function == ConScape.expected_cost && !isnothing(expected_costs)
        # workspace1 .= expected_costs
        # workspace1
        copy(expected_costs)
    elseif connectivity_function == ConScape.free_energy_distance && !isnothing(free_energy_distances)
        workspace1 .= free_energy_distances
        workspace1
    else
        connectivity_function(grsp; kw...)
    end
    # proximities = connectivity_function(grsp; kw...)

    if connectivity_function <: DistanceFunction
        map!(distance_transformation, proximities, proximities)
    end

    maybe_set_diagonal!(S, diagvalue, g.targetnodes)

    # quality scaled proximity matrix
    qSq = workspace2 .= g.qs .* S .* g.qt'

    # square submatrix defined by extracting the rows corresponding to landmarks
    qSq₀₀ = view(workspace3, 1:size(workspace3, 2), :)
    qSq₀₀ .= view(qSq, g.targetnodes, :)

    # size of the full problem
    n = size(g.affinities, 1)

    # node ids for the non-landmarks
    p₁ = setdiff(1:n, g.targetnodes)

    # use an Arnoldi based eigensolver to compute the largest (absolute) eigenvalue and right vector (of submatrix)
    Fps = partialschur(qSq₀₀, nev=1, tol=tol)
    λ₀, vʳ₀ = partialeigen(Fps[1])

    # Some notes on handling intended or unintended landmarks. When the Grid includes landmarks,
    # the proximity matrix is no longer square since columns corresponding to non-landmarks are
    # zero and have been removed. If we denote the full (and therefore square) quality scaled
    # proximity matrix Sq then the rectangular landmark proximity matrix can be written as Sq*P₀
    # where P=(P₀ P₁) is a permutation matrix where P₁ moves all the zero columns to the end. The
    # act of the matrix P₀ correspond to indexing with the vector `targetnodes`.
    #
    # We'd like compute the largest eigen value of Sq but we only have Sq*P₀ and would like to
    # avoid constructing the full Sq if possible. I.e. we'd like to solve |Sq - λI| == 0 witout
    # constructing Sq. Since P is a permutation matrix, |Sq - λ*I| = |P'*Sq*P - λI| and we can
    # expand to
    #                   |/ P₀'*Sq*P₀   P₀'*Sq*P₁ \     |   | / P₀'*Sq*P₀ - λI   0   \|
    # |P'*Sq*P - λ*I| = ||                       | - λI| = | |                      ||
    #                   |\ P₁'*Sq*P₀   P₀'*Sq*P₁ /     |   | \    P₁'*Sq*P₀    -λI  /|
    #
    # since Sq*P₁ = 0. If Sq is n x n and P₀ is n x k then the expressions above show that
    # |Sq - λ*I| == 0 has n - k zero roots and that the non-zero roots are the same as the roots
    # of |P₀'*Sq*P₀ - λI| == 0. Hence we can compute the largest eigenvalue of Sq simply by
    # computing the largest eigenvalue of P₀'*Sq*P₀.
    #
    # To compute the corresponding left and right vectors, we can rewrite the defitions of the
    # left and right eigenvalue problem. Starting the right (usual) right problem
    #
    # Sq*v        = v*λ
    #
    # P'*Sq*P*P'v = P'*v*λ
    #
    # P'*Sq*P*ṽ   = ṽ*λ
    #
    # where ṽ = P'*v. We can again expand the blocks to get
    #
    #  / P₀'*Sq*P₀   0 \/ ṽ₀ \   / ṽ₀ \
    #  |               ||    | = |    |*λ
    #  \ P₁'*Sq*P₀   0 /\ ṽ₁ /   \ ṽ₁ /
    #
    # / P₀'*Sq*P₀*ṽ₀ \   / ṽ₀*λ \
    # |              | = |      |
    # \ P₁'*Sq*P₀*ṽ₀ /   \ ṽ₁*λ /
    #                                                                 P₁'*Sq*P₀*ṽ₀
    # which shows the ṽ₀ is just an eigenvector of P₀'*Sq*P₀ and ṽ₁ = ------------
    #                                                                       λ
    # For the left problem Sq'*w = w*λ, similar calculations leads to
    #
    # / (Sq*P₀)'*P₀*w̃₀ + (Sq*P₀)*P₁'*w̃₁ = w̃₀*λ \
    # |                                        |
    # \              0                  = w̃₁*λ /
    #
    # which shows that w̃₀ is simply a left eigenvector of P₀'*Sq*P₀ and w̃₁ = 0.

    # construct full right vector
    vʳ = fill(NaN, n)
    vʳ[g.targetnodes] = vʳ₀
    vʳ[p₁] = view(qSq, p₁, :) * vʳ₀ / λ₀[1]

    # compute left vector (of submatrix) by shift-invert
    Flu = lu(qSq₀₀ - λ₀[1] * I)
    vˡ₀ = ldiv!(Flu', rand(length(g.targetidx)))
    rmul!(vˡ₀, inv(vˡ₀[1]))

    # construct full left vector
    vˡ = zeros(n)
    vˡ[g.targetnodes] = vˡ₀

    return (vˡ, λ₀=λ₀[1], vʳ)
end

"""
    criticality(grsp::GridRSP[;
                distance_transformation=inv(grsp.g.costfunction),
                diagvalue=nothing,
                avalue=floatmin(),
                qˢvalue=0.0,
                qᵗvalue=0.0])

Compute the landscape criticality for each target cell by setting setting affinities
for the cell to `avalue` as well as the source and target qualities associated with
the cell to `qˢvalue` and `qᵗvalue` respectively. It is required that `avalue` is
positive to avoid that the graph becomes disconnected.
"""
function criticality(grsp::Union{GridRSP,NamedTuple};
    distance_transformation=nothing,
    diagvalue=nothing,
    avalue=floatmin(),
    output=_init_output(grsp.g),
    qˢvalue=0.0,
    qᵗvalue=0.0,
    kw...
)
    g = grsp.g
    nl = length(g.targetidx)
    reference_connected_habitat = sum(connected_habitat(grsp;
        distance_transformation, diagvalue, kw...
    ))
    critvec = fill(reference_connected_habitat, nl)

    @progress name = "Computing criticality..." for i in 1:nl
        critvec[i] = sum(connected_habitat(grsp, g.targetidx[i];
            distance_transformation, diagvalue, avalue, qˢvalue, qᵗvalue, kw...
        ))
    end

    output[g.targetidx] = critvec

    return _maybe_raster(output, grsp)
end

function _init_output(g::Grid)
    o = fill(eltype(g.affinities)(0.0), size(g))
    o[g.id_to_grid_coordinate_list] .= 0
    return o
end
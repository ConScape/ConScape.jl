struct GridRSP
    g::Grid
    θ::Float64
    Pref::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    Z::Matrix{Float64}
end

"""
    GridRSP(g::Grid; θ=nothing)::GridRSP

Construct a GridRSP from a `g::Grid` based on the inverse temperature parameter `θ::Real`.
"""
function GridRSP(g::Grid; θ=nothing)

    Pref = _Pref(g.affinities)
    W    = _W(Pref, θ, g.costmatrix)

    @debug("Computing fundamental matrix of non-absorbing paths (Z). Please be patient...")
    targetidx, targetnodes = _targetidx_and_nodes(g)
    Z    = (I - W)\Matrix(sparse(targetnodes,
                                 1:length(targetnodes),
                                 1.0,
                                 size(g.costmatrix, 1),
                                 length(targetnodes)))
    # Check that values in Z are not too small:
    if minimum(Z)*minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end

    return GridRSP(g, θ, Pref, W, Z)
end

function Base.show(io::IO, ::MIME"text/plain", grsp::GridRSP)
    print(io, summary(grsp), " of size ", grsp.g.nrows, "x", grsp.g.ncols)
end

function Base.show(io::IO, ::MIME"text/html", grsp::GridRSP)
    t = string(summary(grsp), " of size ", grsp.g.nrows, "x", grsp.g.ncols)
    write(io, "<h4>$t</h4>")
    show(io, MIME"text/html"(), plot_outdegrees(grsp.g))
end

"""
    betweenness_qweighted(grsp::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all nodes weighted by source and target qualities.
"""
function betweenness_qweighted(grsp::GridRSP)

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    betvec = RSP_betweenness_qweighted(
        grsp.W,
        grsp.Z,
        [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list],
        [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx],
        targetnodes)

    bet = fill(NaN, grsp.g.nrows, grsp.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[grsp.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end



"""
    edge_betweenness_qweighted(grsp::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all edges weighted by source and target qualities. Returns a
sparse matrix where element (i,j) is the betweenness of edge (i,j).
"""
function edge_betweenness_qweighted(grsp::GridRSP)

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    betmatrix = RSP_edge_betweenness_qweighted(
        grsp.W,
        grsp.Z,
        [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list],
        [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx],
        targetnodes)

    return betmatrix
end


"""
    betweenness_kweighted(grsp::GridRSP;
        connectivity_function=expected_cost,
        distance_transformation=inv(grsp.g.costfunction),
        diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

Compute RSP betweenness of all nodes weighted with proximities computed with respect to the distance/proximity measure defined by `connectivity_function`. Optionally, an inverse cost function can be passed. The function will be applied elementwise to the matrix of distances to convert it to a matrix of proximities. If no inverse cost function is passed the the inverse of the cost function is used for the conversion of distances.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix of proximities, i.e. after applying the inverse cost function to the matrix of distances. When nothing is specified, the diagonal elements won't be adjusted.
"""
function betweenness_kweighted(grsp::GridRSP;
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing)

    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        if grsp.g.costfunction === nothing
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function."))
        else
            distance_transformation = inv(grsp.g.costfunction)
        end
    end

    proximities = connectivity_function(grsp)

    if connectivity_function <: DistanceFunction
        map!(distance_transformation, proximities, proximities)
    end

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            proximities[i, j] = diagvalue
        end
    end

    betvec = RSP_betweenness_kweighted(
        grsp.W,
        grsp.Z,
        [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list],
        [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx],
        proximities,
        targetnodes)

    bet = fill(NaN, grsp.g.nrows, grsp.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[grsp.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end



"""
    edge_betweenness_kweighted(grsp::GridRSP; [distance_transformation=inv(grsp.g.costfunction), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

    Compute RSP betweenness of all edges weighted by qualities of source s and target t and the proximity between s and t. Returns a
    sparse matrix where element (i,j) is the betweenness of edge (i,j).

    The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
    of proximities, i.e. after applying the inverse cost function to the matrix of expected costs.
    When nothing is specified, the diagonal elements won't be adjusted.
"""
function edge_betweenness_kweighted(grsp::GridRSP; distance_transformation=inv(grsp.g.costfunction), diagvalue=nothing)

    proximities = map(distance_transformation, expected_cost(grsp))
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            proximities[i, j] = diagvalue
        end
    end


    betmatrix = RSP_edge_betweenness_kweighted(
        grsp.W,
        grsp.Z,
        [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list],
        [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx],
        proximities,
        targetnodes)

    return betmatrix
end



"""
    expected_cost(grsp::GridRSP)::Matrix{Float64}

Compute RSP expected costs from all nodes.
"""
function expected_cost(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_expected_cost(grsp.W, grsp.g.costmatrix, grsp.Z, targetnodes)
end

function free_energy_distance(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_free_energy_distance(grsp.Z, grsp.θ, targetnodes)
end

function survival_probability(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_survival_probability(grsp.Z, grsp.θ, targetnodes)
end

function power_mean_proximity(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_power_mean_proximity(grsp.Z, grsp.θ, targetnodes)
end

least_cost_distance(grsp::GridRSP) = least_cost_distance(grsp.g)

"""
    mean_kl_divergence(grsp::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP expected costs for `grsp::GridRSP`.
"""
function mean_kl_divergence(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    qs = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qt = [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx]
    return qs'*(RSP_free_energy_distance(grsp.Z, grsp.θ, targetnodes) - expected_cost(grsp))*qt*grsp.θ
end


"""
    mean_lc_kl_divergence(grsp::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the least-cost path and the random path distribution for `grsp::GridRSP`, weighted by the qualities of the source and target node.
"""
function mean_lc_kl_divergence(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    div = hcat([least_cost_kl_divergence(grsp.g.costmatrix, grsp.Pref, i) for i in targetnodes]...)
    qs = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qt = [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx]
    return qs'*div*qt
end

function least_cost_kl_divergence(C::SparseMatrixCSC, Pref::SparseMatrixCSC, targetnode::Integer)

    n = size(C, 1)
    graph = SimpleWeightedDiGraph(C)
    if !(1 <= targetnode <= n)
        throw(ArgumentError("target node not found"))
    end

    dsp = dijkstra_shortest_paths(graph, targetnode)
    parents = dsp.parents
    parents[targetnode] = targetnode

    from = collect(1:n)
    to   = copy(parents)

    kl_div = zeros(n)

    while true
        notdone = false

        for i in 1:n
            fromᵢ = from[i]
            toᵢ   = to[i]
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
        tmp  = from
        from = to
        to   = tmp
    end

    return kl_div
end

"""
    least_cost_kl_divergence(grsp::GridRSP, target::Tuple{Int,Int})

Compute the least cost Kullback-Leibler divergence from each cell in the g in
`h` to the `target` cell.
"""
function least_cost_kl_divergence(grsp::GridRSP, target::Tuple{Int,Int})

    targetnode = findfirst(isequal(CartesianIndex(target)), grsp.g.id_to_grid_coordinate_list)
    if targetnode === nothing
        throw(ArgumentError("target cell not found"))
    end

    div = least_cost_kl_divergence(grsp.g.costmatrix, grsp.Pref, targetnode)

    return reshape(div, grsp.g.nrows, grsp.g.ncols)
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
    grsp::Union{Grid,GridRSP};
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing,
    θ::Union{Nothing,Real}=nothing,
    approx::Bool=false)

    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        if grsp isa Grid
            throw(ArgumentError("distance_transformation function is required when passing a Grid together with a Distance function"))
        elseif grsp.g.costfunction === nothing
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function."))
        else
            distance_transformation = inv(grsp.g.costfunction)
        end
    end

    S = if grsp isa Grid
        if θ === nothing && connectivity_function !== least_cost_distance
            throw(ArgumentError("θ must be a positive real number when passing a Grid"))
        end
        connectivity_function(grsp; θ=θ, approx=approx)
    else
        if θ !== nothing
            throw(ArgumentError("θ must be unspecified when passing a GridRSP"))
        end
        connectivity_function(grsp)
    end

    if connectivity_function <: DistanceFunction
        map!(distance_transformation, S, S)
    end

    return connected_habitat(grsp, S, diagvalue=diagvalue)
end

_get_grid(grsp::GridRSP) = grsp.g
_get_grid(g::Grid)       = g
function connected_habitat(grsp::Union{Grid,GridRSP}, S::Matrix; diagvalue::Union{Nothing,Real}=nothing)

    g = _get_grid(grsp)
    targetidx, targetnodes = _targetidx_and_nodes(g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            S[i, j] = diagvalue
        end
    end

    qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
    qᵗ = [g.target_qualities[i] for i in targetidx]

    funvec = connected_habitat(qˢ, qᵗ, S)

    func = fill(NaN, g.nrows, g.ncols)
    for (ij, x) in zip(g.id_to_grid_coordinate_list, funvec)
        func[ij] = x
    end

    return func
end

function connected_habitat(grsp::GridRSP,
                           cell::CartesianIndex{2};
                           distance_transformation=nothing,
                           diagvalue=nothing,
                           avalue=floatmin(), # smallest non-zero value
                           qˢvalue=0.0,
                           qᵗvalue=0.0)

    if avalue <= 0.0
        throw("Affinity value has to be positive. Otherwise the graph will become disconnected.")
    end

    # Compute (linear) node indices from (cartesian) grid indices
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    node = findfirst(isequal(cell), grsp.g.id_to_grid_coordinate_list)

    # Check that cell is in targetidx
    if cell ∉ targetidx
        throw(ArgumentError("Computing adjusted connected_habitat is only supported for target cells"))
    end

    affinities = copy(grsp.g.affinities)
    affinities[:, node] .= ifelse.(iszero.(affinities[:, node]), 0, avalue)
    affinities[node, :] .= ifelse.(iszero.(affinities[node, :]), 0, avalue)

    newsource_qualities = copy(grsp.g.source_qualities)
    newsource_qualities[cell] = qˢvalue
    newtarget_qualities = copy(grsp.g.target_qualities)
    newtarget_qualities[cell] = qᵗvalue

    newg = Grid(grsp.g.nrows,
                grsp.g.ncols,
                affinities,
                grsp.g.costfunction,
                grsp.g.costfunction === nothing ? grsp.g.costmatrix : mapnz(grsp.g.costfunction, affinities),
                grsp.g.id_to_grid_coordinate_list,
                newsource_qualities,
                newtarget_qualities)

    newh = GridRSP(newg, θ=grsp.θ)

    return connected_habitat(newh; diagvalue=diagvalue, distance_transformation=distance_transformation)
end

"""
    eigmax(grsp::GridRSP;
        connectivity_function=expected_cost,
        distance_transformation=nothing,
        diagvalue=nothing,
        tol=1e-14)

Compute the largest eigenvalue triple (left vector, value, and right vector) of the quality scaled proximities with respect to the distance/proximity measure defined by `connectivity_function`. If `connectivity_function` is a distance measure then the distances are transformed to proximities by `distance_transformation` which defaults to the inverse of the `costfunction` in the underlying `Grid` (if defined). Optionally, the diagonal values of the proximity matrix may be set to `diagvalue`. The `tol` argument specifies the convergence tolerance in the Arnoldi based eigensolver.
"""
function LinearAlgebra.eigmax(grsp::GridRSP;
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing,
    tol=1e-14)

    g = grsp.g

    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        if grsp.g.costfunction === nothing
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function."))
        else
            distance_transformation = inv(grsp.g.costfunction)
        end
    end

    S = connectivity_function(grsp)

    if connectivity_function <: DistanceFunction
        map!(distance_transformation, S, S)
    end

    targetidx, targetnodes = _targetidx_and_nodes(g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            S[i, j] = diagvalue
        end
    end

    qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
    qᵗ = [g.target_qualities[i] for i in targetidx]

    # quality scaled proximity matrix
    qSq = qˢ .* S .* qᵗ'

    # square submatrix defined by extracting the rows corresponding to landmarks
    qSq₀₀ = qSq[targetnodes,:]

    # size of the full problem
    n = size(g.affinities, 1)

    # node ids for the non-landmarks
    p₁ = setdiff(1:n, targetnodes)

    # use an Arnoldi based eigensolver to compute the largest (absolute) eigenvalue and right vector (of submatrix)
    Fps     = partialschur(qSq₀₀, nev=1, tol=tol)
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
    vʳ[targetnodes] = vʳ₀
    vʳ[p₁] = qSq[p₁,:]*vʳ₀/λ₀[1]

    # compute left vector (of submatrix) by shift-invert
    Flu = lu(qSq₀₀ - λ₀[1]*I)
    vˡ₀ = ldiv!(Flu', rand(length(targetidx)))
    rmul!(vˡ₀, inv(vˡ₀[1]))

    # construct full left vector
    vˡ = zeros(n)
    vˡ[targetnodes] = vˡ₀

    return vˡ, λ₀[1], vʳ
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
function criticality(grsp::GridRSP;
                     distance_transformation=nothing,
                     diagvalue=nothing,
                     avalue=floatmin(),
                     qˢvalue=0.0,
                     qᵗvalue=0.0)

    targetidx, _ = _targetidx_and_nodes(grsp.g)
    nl = length(targetidx)
    reference_connected_habitat = sum(connected_habitat(grsp, distance_transformation=distance_transformation, diagvalue=diagvalue))
    critvec = fill(reference_connected_habitat, nl)

    @progress name="Computing criticality..." for i in 1:nl
        critvec[i] -= sum(connected_habitat(
            grsp,
            targetidx[i];
            distance_transformation=distance_transformation,
            diagvalue=diagvalue,
            avalue=avalue,
            qˢvalue=qˢvalue,
            qᵗvalue=qᵗvalue))
    end

    landscape = fill(NaN, size(grsp.g))
    landscape[targetidx] = critvec

    return landscape
end


"""
    sensitivity(grsp::ConScape.GridRSP;
        connectivity_function=ConScape.expected_cost,
        distance_transformation=ConScape.ExpMinus(),
        α::Union{Nothing,Real}=1.,
        wrt::String=["A", "C", "Q","C&A=f(C)", "A&C=f(A)"][1],
        landscape_measure::String=["sum","eigenanalysis"][1],
        unitless::Bool=true,
        diagvalue=nothing,
        target_equal_source::Bool=true)::Matrix{Float64}

Compute sensitivity of all nodes. Five types of node sensitivity are implemented, 
two different `landscape_measures` are implemented to summarize the landscape matrix 
either through summation or through eigenanalysis. The results can be provided either as 
sensitivity w.r.t. unit change (`unitless`=false) or w.r.t. proportional change (`unitless`=true), 
the latter are also known as elasticities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `distance_transformation`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`. 
α is a distance scaling, and it is multiplied with the distance
"""
function sensitivity(grsp::ConScape.GridRSP;
    connectivity_function=ConScape.expected_cost,
    distance_transformation=ConScape.ExpMinus(),
    α::Union{Nothing,Real}=1.,
    wrt::String=["A", "C", "Q","C&A=f(C)", "A&C=f(A)"][1],
    landscape_measure::String=["sum","eigenanalysis"][1],
    unitless::Bool=true,
    diagvalue=nothing,
    target_equal_source::Bool=true)

    if (!(wrt in ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]))
        throw(ArgumentError("sensitivity is only defined with respect to (wrt) affinities, costs, or pixel qualities"))
    end

    #throw(ArgumentError("Got here"))

    if (!(landscape_measure in ["sum","eigenanalysis"]))
        throw(ArgumentError("The landscape needs to be summarized either as the sum of the landscape matrix or its leading eigenvalue"))
    end

    if (!(target_equal_source))
        throw(ArgumentError("The sensitivity is currently only defined when target quality equals source quality."))
    end

    if distance_transformation !== ConScape.ExpMinus() && α !== 1.
        throw(ArgumentError("α can different from 1 only when using exponential proximity transformation"))
    end

    if connectivity_function === ConScape.survival_probability && grsp.θ !==1.0
        throw(ArgumentError("The survival probability is currently only implemented for theta = 1."))
    end

    if (distance_transformation === ConScape.ExpMinus())
        distance_transformation_alpha=x -> exp(-x*α)
    elseif (distance_transformation === ConScape.Inv())
        distance_transformation_alpha=x -> inv(x)
    elseif distance_transformation === nothing    # Check that distance_transformation function has been passed. If not then default to inverse of cost function
        distance_transformation_alpha = inv(grsp.g.costfunction)
    #else
    #    throw(ArgumentError("The sensitivity is currently only defined for negative exponential and inverse distance transformations."))
    end

    if grsp.g.costfunction === nothing && (wrt in ["C&A=f(C)", "A&C=f(A)"])
        throw(ArgumentError("C&A=f(C) or A&C=f(A) sensitivities are only defined when costs are defined as functions of affinities"))
    end

    targetidx, targetnodes = ConScape._targetidx_and_nodes(grsp.g)
    qˢ = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qᵗ = [grsp.g.target_qualities[i] for i in targetidx]

    if (landscape_measure === "eigenanalysis")
        v, λ, w = ConScape.eigmax(grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha)
        vTw = (v') * w
        
        if (wrt !== "Q")
            qˢ = qˢ .* v    
            qᵗ = qᵗ .* w
        end
    end

    if (wrt in ["A&C=f(A)", "C&A=f(C)"]) 
        # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
        # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij
        if grsp.g.costfunction == ConScape.MinusLog()
            diff_C_A_fun = x -> -inv(x)
            diff_A_C_fun = x -> -(x)
        elseif grsp.g.costfunction == ConScape.Inv()
            diff_C_A_fun = x -> -inv(x^2)
            diff_A_C_fun = x -> -inv(x^2)
        elseif grsp.g.costfunction === nothing
            diff_C_A_fun = x -> -inv(x)
            diff_A_C_fun = x -> -(x)
            @info "Cost function is nothing, so for the linked sensitivity the minuslog link will be used"
        end
    end

    if (wrt in ["A", "C", "A&C=f(A)", "C&A=f(C)"]) 
        if connectivity_function === ConScape.expected_cost
                # Derivative of costs w.r.t. affinities:
            # TODO: Implement these as properties of Transformations:
            K = map(distance_transformation_alpha, ConScape.expected_cost(grsp))

            if diagvalue !== nothing
                for (j, i) in enumerate(targetnodes)
                    K[i, j] = diagvalue
                end
            end

            K[isinf.(K)] .= 0.

            if (distance_transformation === ConScape.ExpMinus())
                diff_K_D = -K .* α
            elseif (distance_transformation === ConScape.Inv())
                diff_K_D = -K.^2
            elseif distance_transformation === nothing    # Check that distance_transformation function has been passed. If not then default to negative exponential
                diff_K_D = -K .* α
            #else
            #    throw(ArgumentError("The sensitivity is currently only defined for negative exponential and inverse distance transformations."))
            end

            S_e_aff, S_e_cost = EC_sensitivity(
                grsp.g.affinities,
                grsp.g.costmatrix,
                grsp.θ,
                grsp.W,
                grsp.Z,
                diff_K_D,
                qˢ,
                qᵗ,
                targetnodes)

            if unitless && (wrt in ["A", "A&C=f(A)"])
                S_e_aff = S_e_aff.*grsp.g.affinities
                S_e_cost = S_e_cost.*grsp.g.affinities
            elseif unitless && (wrt in ["C", "C&A=f(C)"])
                S_e_aff = S_e_aff.*grsp.g.costmatrix
                S_e_cost = S_e_cost.*grsp.g.costmatrix
            end

            if (wrt in ["A"])
                node_sensitivity_vec = vec(sum(S_e_aff, dims=1))
            elseif (wrt in ["C"])
                node_sensitivity_vec = vec(sum(S_e_cost, dims=1))
            elseif (wrt in ["A&C=f(A)"])
                diff_C_A = ConScape.mapnz(diff_C_A_fun, grsp.g.affinities)
                S_e_total = S_e_aff .+ S_e_cost.*diff_C_A
                node_sensitivity_vec = vec(sum(S_e_total, dims=1))
            elseif (wrt in ["C&A=f(C)"])
                diff_A_C = ConScape.mapnz(diff_A_C_fun, grsp.g.affinities)
                S_e_total = S_e_aff.*diff_A_C .+ S_e_cost
                node_sensitivity_vec = vec(sum(S_e_total, dims=1))
            end

        elseif connectivity_function !== ConScape.expected_cost
            #for theta = 1, ConScape.power_mean_proximity = ConScape.survival_probability
            # Now assumes grsp.g.costfunction = MinusLog

            S_e_aff, S_e_cost = PM_sensitivity(grsp.g.affinities, nothing, grsp.θ, grsp.W, grsp.Z, grsp.Z, qˢ, qᵗ, targetnodes)

            if unitless && (wrt in ["A", "A&C=f(A)"])
                S_e_aff = S_e_aff.*grsp.g.affinities
                S_e_cost = S_e_cost.*grsp.g.affinities
            elseif unitless && (wrt in ["C", "C&A=f(C)"])
                S_e_aff = S_e_aff.*grsp.g.costmatrix
                S_e_cost = S_e_cost.*grsp.g.costmatrix
            end

            if (wrt in ["A"])
                node_sensitivity_vec = vec(sum(S_e_aff, dims=1))
            elseif (wrt in ["C"])
                node_sensitivity_vec = vec(sum(S_e_cost, dims=1))
            elseif (wrt in ["A&C=f(A)"])
                diff_C_A = ConScape.mapnz(diff_C_A_fun, grsp.g.affinities)
                S_e_total = S_e_aff .+ S_e_cost.*diff_C_A
                node_sensitivity_vec = vec(sum(S_e_total, dims=1))
            elseif (wrt in ["C&A=f(C)"])
                diff_A_C = ConScape.mapnz(diff_A_C_fun, grsp.g.affinities)
                S_e_total = S_e_aff.*diff_A_C .+ S_e_cost
                node_sensitivity_vec = vec(sum(S_e_total, dims=1))
            end
        end
    elseif wrt ==="Q"
        K = connectivity_function(grsp)
        if connectivity_function <: ConScape.DistanceFunction
            map!(distance_transformation_alpha, K, K)
        end
        
        if (landscape_measure === "eigenanalysis")
            K = (v') * K
            K = w * K
        end

        K = K + transpose(K)

        node_sensitivity_vec = K *  grsp.g.target_qualities[grsp.g.id_to_grid_coordinate_list]
        if unitless
            node_sensitivity_vec = node_sensitivity_vec .*  grsp.g.source_qualities[grsp.g.id_to_grid_coordinate_list]
        end
    else
        throw(ArgumentError("Invalid or not implemented combination of arguments"))    
    end

    if (landscape_measure === "eigenanalysis")
        vTw = (v') * w
        node_sensitivity_vec = node_sensitivity_vec ./ (vTw)
    end

    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in grsp.g.id_to_grid_coordinate_list], [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
    node_sensitivity_vec, grsp.g.nrows, grsp.g.ncols)) .* map(x-> isnan(x) ? NaN : 1, grsp.g.source_qualities)

    return node_sensitivity_matrix

end


"""
    criticality_simulation(grsp::ConScape.GridRSP;
        connectivity_function=ConScape.expected_cost,
        distance_transformation=ConScape.ExpMinus(),
        α::Union{Nothing,Real}=1.,
        wrt::String=["all", "Q"][1],
        landscape_measure::String=["sum","eigenanalysis"][1],
        diagvalue=nothing, 
        target_equal_source::Bool=true, 
        one_out_of::Int64=1)::Matrix{Float64}

Compute criticality of all nodes. Two types of node criticality are implemented: 
(1) wrt="all" completely destroys a node from the graph, whereas 
(2) wrt="Q" only removes the quality of a node, but leaves the node as a connector in the Grid.

Two different `landscape_measures` are implemented to summarize the landscape matrix 
either through summation or through eigenanalysis. 

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `distance_transformation`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`. 
α is a distance scaling, and it is multiplied with the distance

The function iteratively removes each node, which takes a long time. To speed up computation, 
    we added a subsampling option, where only `one_out_of` every so many pixels is evaluated.
"""
function criticality_simulation(grsp::ConScape.GridRSP;
    connectivity_function=ConScape.expected_cost,
    distance_transformation=ConScape.ExpMinus(),
    α::Union{Nothing,Real}=1.,
    wrt::String=["all", "Q"][1],
    landscape_measure::String=["sum","eigenanalysis"][1],
    diagvalue=nothing, 
    target_equal_source::Bool=true, 
    one_out_of::Int64=1)

    if (!(wrt in ["all", "Q"]))
        throw(ArgumentError("Criticality is only defined with respect to (wrt) connectivity and qualities (i.e. affinities, cost and qualities; all), or pixel qualities (Q)"))
    end

    if (!(landscape_measure in ["sum","eigenanalysis"]))
        throw(ArgumentError("The landscape needs to be summarized either as the sum of the landscape matrix or its leading eigenvalue"))
    end

    if (!(target_equal_source))
        throw(ArgumentError("The criticality is currently only defined when target quality equals source quality."))
    end

    if (distance_transformation==ConScape.ExpMinus())
        distance_transformation_alpha=x -> exp(-x*α)
    elseif (distance_transformation==ConScape.Inv())
        distance_transformation_alpha=x -> inv(x*α)
    #else
    #    throw(ArgumentError("The sensitivity is currently only defined for negative exponential and inverse distance transformations."))
    end

    old_g = ConScape.Grid(size(grsp.g)...,
    grsp.g.affinities,
    nothing,
    grsp.g.costmatrix,
    grsp.g.id_to_grid_coordinate_list,
    grsp.g.source_qualities,
    grsp.g.source_qualities)

    old_grsp = ConScape.GridRSP(old_g, θ=grsp.θ)
    lf = ConScape.connected_habitat(old_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
    replace!(x -> isnan(x) ? 0 : x, lf)

    n = length(grsp.g.id_to_grid_coordinate_list)
    node_sensitivities = repeat([0.0], n)

    if (wrt === "all")
        for i in 1:n
            if (i % one_out_of == 0)
                new_affinities = copy(grsp.g.affinities)
                new_affinities = new_affinities[(1:end) .!= i, (1:end) .!= i]
                new_costs = copy(grsp.g.costmatrix)
                new_costs = new_costs[(1:end) .!= i, (1:end) .!= i]
                new_id_to_grid_coordinate_list = copy(grsp.g.id_to_grid_coordinate_list)

                new_g = ConScape.Grid(size(grsp.g)...,
                    new_affinities,
                    nothing,
                    new_costs,
                    deleteat!(new_id_to_grid_coordinate_list, i),
                    grsp.g.source_qualities,
                    grsp.g.target_qualities)           

                new_grsp = ConScape.GridRSP(new_g, θ=grsp.θ)
                new_lf = ConScape.connected_habitat(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                replace!(x -> isnan(x) ? 0 : x, new_lf)

                node_sensitivities[i] = sum(new_lf - lf)
            else
                node_sensitivities[i] = NaN
            end
        end
    elseif (wrt === "Q")
        for i in 1:n
            if (i % one_out_of == 0)

                new_qualities = copy(grsp.g.source_qualities)
                new_qualities[grsp.g.id_to_grid_coordinate_list[i]] = 0

                new_g = ConScape.Grid(size(grsp.g)...,
                    grsp.g.affinities,
                    nothing,
                    grsp.g.costmatrix,
                    grsp.g.id_to_grid_coordinate_list,
                    new_qualities,
                    new_qualities)

                new_grsp = ConScape.GridRSP(new_g, θ=grsp.θ)
                new_lf = ConScape.connected_habitat(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                replace!(x -> isnan(x) ? 0 : x, new_lf)

                node_sensitivities[i] = sum(new_lf - lf)
            else
                node_sensitivities[i] = NaN
            end
        end
    end

    return Matrix(sparse(
        [ij[1] for ij in grsp.g.id_to_grid_coordinate_list],
        [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
        node_sensitivities,
        grsp.g.nrows,
        grsp.g.ncols)) .* map(x-> isnan(x) ? NaN : 1, grsp.g.source_qualities)
end


"""
    sensitivity_similation(grsp::ConScape.GridRSP;
        connectivity_function=ConScape.expected_cost,
        distance_transformation=ConScape.ExpMinus(),
        α::Union{Nothing,Real}=1.,
        wrt::String=["A", "C", "Q","C&A=f(C)", "A&C=f(A)"][1],
        landscape_measure::String=["sum","eigenanalysis"][1],
        unitless::Bool=true,
        diagvalue=nothing,
        target_equal_source::Bool=true)::Matrix{Float64}

Compute sensitivity of all nodes. Five types of node sensitivity are implemented, 
two different `landscape_measures` are implemented to summarize the landscape matrix 
either through summation or through eigenanalysis. The results can be provided either as 
sensitivity w.r.t. unit change (`unitless`=false) or w.r.t. proportional change (`unitless`=true), 
the latter are also known as elasticities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `distance_transformation`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`. 
α is a distance scaling, and it is multiplied with the distance

The function iteratively perturbates each node, which takes a long time. To speed up computation, 
    we added a subsampling option, where only `one_out_of` every so many pixels is evaluated.
"""
function sensitivity_simulation(grsp::ConScape.GridRSP;
    connectivity_function=ConScape.expected_cost,
    distance_transformation=ConScape.ExpMinus(),
    α::Union{Nothing,Real}=1.,
    wrt::String=["A", "C", "Q","C&A=f(C)", "A&C=f(A)"][1],
    landscape_measure::String=["sum","eigenanalysis"][1],
    unitless::Bool=true,
    diagvalue=nothing, 
    target_equal_source::Bool=true, 
    one_out_of::Int64=1)

    if (!(wrt in ["A", "C", "Q","A&C=f(A)", "C&A=f(C)"]))
        throw(ArgumentError("sensitivity is only defined with respect to (wrt) affinities, costs, or pixel qualities"))
    end

    if (!(landscape_measure in ["sum","eigenanalysis"]))
        throw(ArgumentError("The landscape needs to be summarized either as the sum of the landscape matrix or its leading eigenvalue"))
    end

    if (!(target_equal_source))
        throw(ArgumentError("The sensitivity is currently only defined when target quality equals source quality."))
    end

    if (distance_transformation==ConScape.ExpMinus())
        distance_transformation_alpha=x -> exp(-x*α)
    elseif (distance_transformation==ConScape.Inv())
        distance_transformation_alpha=x -> inv(x*α)
    #else
    #    throw(ArgumentError("The sensitivity is currently only defined for negative exponential and inverse distance transformations."))
    end

    epsi = 1e-6

    if (landscape_measure === "eigenanalysis")
        v, lf, w = ConScape.eigmax(grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
    else
        lf = ConScape.connected_habitat(grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
        replace!(x -> isnan(x) ? 0 : x, lf)    
    end

    n = length(grsp.g.id_to_grid_coordinate_list)

    if (wrt in ["A", "C", "A&C=f(A)", "C&A=f(C)"])

        edge_sensitivities = copy(grsp.g.affinities)

        for i in 1:n
            Succ_i = findall(grsp.g.affinities[i,:].>0)
            for j in Succ_i
                if (j % one_out_of == 0)
                    if (wrt in ["A"])
                        new_affinities = copy(grsp.g.affinities)
                        new_costs = copy(grsp.g.costmatrix)
                        new_affinities[i, j] += epsi
                    elseif (wrt in ["C"])
                        new_affinities = copy(grsp.g.affinities)
                        new_costs = copy(grsp.g.costmatrix)
                        new_costs[i, j] += epsi
                    elseif (wrt in ["A&C=f(A)"])
                        new_affinities = copy(grsp.g.affinities)
                        new_affinities[i, j] += epsi
                        if grsp.g.costfunction === nothing
                            new_costs = ConScape.mapnz(ConScape.MinusLog(), new_affinities)
                            if i === 1
                                @info "Cost function is nothing, so for the linked sensitivity the minuslog link will be used" 
                            end 
                        else
                            new_costs = ConScape.mapnz(grsp.g.costfunction, new_affinities)
                        end
                    elseif (wrt in ["C&A=f(C)"])
                        new_costs = copy(grsp.g.costmatrix)
                        new_costs[i, j] += epsi
                        new_affinities = copy(new_costs)
                        if grsp.g.costfunction === nothing
                            map!(inv(ConScape.MinusLog()), new_affinities.nzval, new_costs.nzval)
                            if i === 1
                                @info "Cost function is nothing, so for the linked sensitivity the minuslog link will be used" 
                            end 
                        else
                            map!(inv(grsp.g.costfunction), new_affinities.nzval, new_costs.nzval)
                        end
                    end

                    new_g = ConScape.Grid(size(grsp.g)...,
                        new_affinities,
                        nothing,
                        new_costs,
                        grsp.g.id_to_grid_coordinate_list,
                        grsp.g.source_qualities,
                        grsp.g.target_qualities)           

                    new_grsp = ConScape.GridRSP(new_g, θ=grsp.θ)

                    if (landscape_measure === "eigenanalysis")
                        v, new_lf, w = ConScape.eigmax(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                    else
                        new_lf = ConScape.connected_habitat(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                        replace!(x -> isnan(x) ? 0 : x, new_lf)
                    end

                    edge_sensitivities[i,j] = sum(new_lf - lf)/epsi # (gnew.affinities[i,j]-g.affinities[i,j])
                    if unitless
                        if (wrt in ["A", "A&C=f(A)"])
                            edge_sensitivities[i,j] *=grsp.g.affinities[i,j]
                        else
                            edge_sensitivities[i,j] *=grsp.g.costmatrix[i,j]
                        end
                    end
                else
                    edge_sensitivities[i,j] = NaN
                end

            end
        end

        node_sensitivities = vec(sum(edge_sensitivities, dims=1))

    elseif (wrt == "Q")
        node_sensitivities = repeat([0.0], n)

        #Some more debugging is needed, but it works with this "patch" 
        old_g = ConScape.Grid(size(grsp.g)...,
                    grsp.g.affinities,
                    nothing,
                    grsp.g.costmatrix,
                    grsp.g.id_to_grid_coordinate_list,
                    grsp.g.source_qualities,
                    grsp.g.source_qualities)

        old_grsp = ConScape.GridRSP(old_g, θ=grsp.θ)

        if (landscape_measure === "eigenanalysis")
            v, lf, w = ConScape.eigmax(old_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
        else
            lf = ConScape.connected_habitat(old_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
            replace!(x -> isnan(x) ? 0 : x, lf)
        end

        for i in 1:n
            if (i % one_out_of == 0)

                new_qualities = copy(grsp.g.source_qualities)
                new_qualities[grsp.g.id_to_grid_coordinate_list[i]] += epsi

                new_g = ConScape.Grid(size(grsp.g)...,
                    grsp.g.affinities,
                    nothing,
                    grsp.g.costmatrix,
                    grsp.g.id_to_grid_coordinate_list,
                    new_qualities,
                    new_qualities)

                new_grsp = ConScape.GridRSP(new_g, θ=grsp.θ)

                if (landscape_measure === "eigenanalysis")
                    v, new_lf, w = ConScape.eigmax(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                else
                    new_lf = ConScape.connected_habitat(new_grsp, connectivity_function=connectivity_function, distance_transformation=distance_transformation_alpha, diagvalue=diagvalue)
                    replace!(x -> isnan(x) ? 0 : x, new_lf)
                end
        
                node_sensitivities[i] = sum(new_lf - lf)/epsi
                if unitless
                    node_sensitivities[i] *= grsp.g.source_qualities[grsp.g.id_to_grid_coordinate_list[i]]
                end
            else
                node_sensitivities[i] = NaN
            end
        end
    end

    return Matrix(sparse(
        [ij[1] for ij in grsp.g.id_to_grid_coordinate_list],
        [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
        node_sensitivities,
        grsp.g.nrows,
        grsp.g.ncols)) .* map(x-> isnan(x) ? NaN : 1, grsp.g.source_qualities)
end



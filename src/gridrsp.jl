abstract type ConnectivityFunction <: Function end
abstract type DistanceFunction <: ConnectivityFunction end
abstract type ProximityFunction <: ConnectivityFunction end

struct expected_cost         <: DistanceFunction end
struct free_energy_distance  <: DistanceFunction end

struct survival_probability  <: ProximityFunction end
struct power_mean_proximity  <: ProximityFunction end

struct GridRSP
    g::Grid
    β::Float64
    Pref::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    Z::Matrix{Float64}
end

"""
    GridRSP(g::Grid; β=nothing)::GridRSP

Construct a GridRSP from a `g::Grid` based on the temperature parameter `β::Real`.
"""
function GridRSP(g::Grid; β=nothing)

    Pref = _Pref(g.affinities)
    W    = _W(Pref, β, g.costmatrix)

    @debug("Computing fundamental matrix of non-absorbing paths (Z). Please be patient...")
    targetidx, targetnodes = _targetidx_and_nodes(g)
    Z    = (I - W)\Matrix(sparse(targetnodes,
                                 1:length(targetnodes),
                                 1.0,
                                 size(g.costmatrix, 1),
                                 length(targetnodes)))
    # Check that values in Z are not too small:
    if minimum(Z)*minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing β."
    end

    return GridRSP(g, β, Pref, W, Z)
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
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in Grid isn't based on a cost function."))
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
    return RSP_free_energy_distance(grsp.Z, grsp.β, targetnodes)
end

function survival_probability(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_survival_probability(grsp.Z, grsp.β, targetnodes)
end

function power_mean_proximity(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_power_mean_proximity(grsp.Z, grsp.β, targetnodes)
end

"""
    mean_kl_divergence(grsp::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP expected costs for `grsp::GridRSP`.
"""
function mean_kl_divergence(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    qs = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qt = [grsp.g.target_qualities[i] for i in grsp.g.id_to_grid_coordinate_list ∩ targetidx]
    return qs'*(RSP_free_energy_distance(grsp.Z, grsp.β, targetnodes) - expected_cost(grsp))*qt*grsp.β
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

    # from = LinearIndices((grsp.g.nrows, grsp.g.ncols))[grsp.g.id_to_grid_coordinate_list]
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
            # v = grsp.Pref[toᵢ, fromᵢ]
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
    connected_habitat(grsp::GridRSP;
        connectivity_function=expected_cost,
        distance_transformation=inv(grsp.g.costfunction),
        diagvalue=nothing)::Matrix{Float64}

Compute RSP connected_habitat of all nodes. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the proximities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `distance_transformation`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`.
"""
function connected_habitat(grsp::GridRSP;
                           connectivity_function=expected_cost,
                           distance_transformation=nothing,
                           diagvalue=nothing)

    # Check that distance_transformation function has been passed if no cost function is saved
    if distance_transformation === nothing && connectivity_function <: DistanceFunction
        if grsp.g.costfunction === nothing
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in Grid isn't based on a cost function."))
        else
            distance_transformation = inv(grsp.g.costfunction)
        end
    end

    S = connectivity_function(grsp)

    if connectivity_function <: DistanceFunction
        map!(distance_transformation, S, S)
    end

    return connected_habitat(grsp, S, diagvalue=diagvalue)
end

function connected_habitat(grsp::GridRSP, S::Matrix; diagvalue::Union{Nothing,Real}=nothing)

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            S[i, j] = diagvalue
        end
    end

    qˢ = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qᵗ = [grsp.g.target_qualities[i] for i in targetidx]

    funvec = connected_habitat(qˢ, qᵗ, S)

    func = sparse([ij[1] for ij in grsp.g.id_to_grid_coordinate_list],
                  [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
                  funvec,
                  grsp.g.nrows,
                  grsp.g.ncols)

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

    newh = GridRSP(newg, β=grsp.β)

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
            throw(ArgumentError("no distance_transformation function supplied and cost matrix in Grid isn't based on a cost function."))
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

    @showprogress 1 "Computing criticality..." for i in 1:nl
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

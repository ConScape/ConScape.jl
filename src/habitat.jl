struct Habitat
    g::Grid
    cost::Cost
    β::Float64
    C::SparseMatrixCSC{Float64,Int}
    Pref::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    Z::Matrix{Float64}
end

"""
    Habitat(g::Grid; cost::Cost=MinusLog(), β=nothing)::Habitat

Construct a Habitat from a `g::Grid` based on a `cost::Cost` type and the temperature `β::Real`.
"""
function Habitat(g::Grid; cost::Cost=MinusLog(), β=nothing)
    C    = mapnz(cost, g.A)
    Pref = _Pref(g.A)
    W    = _W(Pref, β, C)
    @info("Compututing fundamental matrix of non-absorbing paths (Z). Please be patient...")
    targetidx = findall(!iszero, g.target_qualities)
    targetnodes = findall(
        t -> t ∈ targetidx,
        g.id_to_grid_coordinate_list)
    Z    = (I - W)\Matrix(sparse(targetnodes,
                                 1:length(targetidx),
                                 1.0,
                                 size(C, 1),
                                 length(targetidx)))
    return Habitat(g, cost, β, C, Pref, W, Z)
end

function Base.show(io::IO, ::MIME"text/plain", h::Habitat)
    print(io, summary(h), " of size ", h.g.nrows, "x", h.g.ncols)
end

function Base.show(io::IO, ::MIME"text/html", h::Habitat)
    t = string(summary(h), " of size ", h.g.nrows, "x", h.g.ncols)
    write(io, "<h4>$t</h4>")
    show(io, MIME"text/html"(), plot_outdegrees(h.g))
end

"""
    RSP_betweenness_qweighted(h::Habitat)::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted by source and target qualities.
"""
function RSP_betweenness_qweighted(h::Habitat)

    targetidx = findall(!iszero, h.g.target_qualities)
    targetnodes = findall(
        t -> t ∈ targetidx,
        h.g.id_to_grid_coordinate_list)

    betvec = RSP_betweenness_qweighted(
        h.W,
        h.Z,
        [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
        [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
        targetnodes)

    bet = fill(NaN, h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end


"""
    RSP_betweenness_kweighted(h::Habitat; [invcost=inv(h.cost)])::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.
"""
function RSP_betweenness_kweighted(h::Habitat; invcost=inv(h.cost))

    similarities = map(t -> iszero(t) ? t : invcost(t), RSP_dissimilarities(h))

    targetidx = findall(!iszero, h.g.target_qualities)
    targetnodes = findall(
        t -> t ∈ targetidx,
        h.g.id_to_grid_coordinate_list)

    betvec = RSP_betweenness_kweighted(h.W,
                                       h.Z,
                                       [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
                                       [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
                                            similarities,
                                            targetnodes)
    bet = fill(NaN, h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end

"""
    RSP_dissimilarities(h::Habitat)::Matrix{Float64}

Compute RSP expected costs or RSP dissimilarities from all nodes.
"""
function RSP_dissimilarities(h::Habitat)
    targetidx = findall(!iszero, h.g.target_qualities)
    targetnodes = findall(
        t -> t ∈ targetidx,
        h.g.id_to_grid_coordinate_list)
    return RSP_dissimilarities(h.W, h.C, h.Z, targetnodes)
end

RSP_free_energy_distance(h::Habitat) = RSP_free_energy_distance(h.Z, h.β)

"""
    mean_kl_divergence(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP dissimilarities for `h::Habitat`.
"""
function mean_kl_divergence(h::Habitat)
    qs = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qt = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    return qs'*(RSP_free_energy_distance(h.Z, h.β) - RSP_dissimilarities(h))*qt*h.β
end


"""
    mean_lc_kl_divergence(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the least-cost path and the random path distribution for `h::Habitat`, weighted by the qualities of the source and target node.
"""
function mean_lc_kl_divergence(h::Habitat)
    div = hcat([least_cost_kl_divergence(h.C, h.Pref, i) for i in 1:size(h.C, 1)]...)
    qs = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qt = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ findall(!iszero, h.g.target_qualities)]
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

    # from = LinearIndices((h.g.nrows, h.g.ncols))[h.g.id_to_grid_coordinate_list]
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
            # v = h.Pref[toᵢ, fromᵢ]
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
    least_cost_kl_divergence(h::Habitat, target::Tuple{Int,Int})

Compute the least cost Kullback-Leibler divergence from each cell in the g in
`h` to the `target` cell.
"""
function least_cost_kl_divergence(h::Habitat, target::Tuple{Int,Int})

    targetnode = findfirst(isequal(CartesianIndex(target)), h.g.id_to_grid_coordinate_list)
    if targetnode === nothing
        throw(ArgumentError("target cell not found"))
    end

    div = least_cost_kl_divergence(h.C, h.Pref, targetnode)

    return reshape(div, h.g.nrows, h.g.ncols)
end

"""
    RSP_functionality(h::Habitat; [invcost=inv(h.cost)])::Matrix{Float64}

Compute RSP functionality of all nodes. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.
"""
function RSP_functionality(h::Habitat; invcost=inv(h.cost))

    S = RSP_dissimilarities(h)
    map!(t -> iszero(t) ? t : invcost(t), S, S)

    targetidx = findall(!iszero, h.g.target_qualities)
    targetnodes = findall(
        t -> t ∈ targetidx,
        h.g.id_to_grid_coordinate_list)

    funvec = RSP_functionality([h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
                               [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
                               S)
    func = fill(NaN, h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(funvec)
        func[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return func
end

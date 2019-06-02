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
    Z    = inv(Matrix(I - W))
    return Habitat(g, cost, β, C, Pref, W, Z)
end

"""
    RSP_full_betweenness_qweighted(h::Habitat)::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted by source and target qualities.
"""
function RSP_full_betweenness_qweighted(h::Habitat)

    betvec = RSP_full_betweenness_qweighted(
        h.Z,
        h.g.source_qualities[h.g.id_to_grid_coordinate_list],
        h.g.target_qualities[h.g.id_to_grid_coordinate_list])

    bet = zeros(h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end


"""
    RSP_full_betweenness_kweighted(h::Habitat; [invcost=inv(h.cost)])::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.
"""
function RSP_full_betweenness_kweighted(h::Habitat; invcost=inv(h.cost))

    similarities = map(t -> iszero(t) ? t : invcost(t), RSP_dissimilarities(h))
    betvec = RSP_full_betweenness_kweighted(h.Z,
                                            h.g.source_qualities[h.g.id_to_grid_coordinate_list],
                                            h.g.target_qualities[h.g.id_to_grid_coordinate_list],
                                            similarities)
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
RSP_dissimilarities(h::Habitat) = RSP_dissimilarities(h.W, h.C, h.Z)

RSP_free_energy_distance(h::Habitat) = RSP_free_energy_distance(h.Z, h.β)

"""
    mean_kl_divergence(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP dissimilarities for `h::Habitat`.
"""
function mean_kl_divergence(h::Habitat)
    qs = h.g.source_qualities[h.g.id_to_grid_coordinate_list]
    qt = h.g.target_qualities[h.g.id_to_grid_coordinate_list]
    return qs'*(RSP_free_energy_distance(h.Z, h.β) - RSP_dissimilarities(h))*qt*h.β
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

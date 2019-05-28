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
    bet = zeros(h.g.nrows, h.g.ncols)
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
    mean_kl_distance(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP dissimilarities for `h::Habitat`.
"""
function mean_kl_distance(h::Habitat)
    qs = h.g.source_qualities[h.g.id_to_grid_coordinate_list]
    qt = h.g.target_qualities[h.g.id_to_grid_coordinate_list]
    return qs'*(RSP_free_energy_distance(h.Z, h.β) - RSP_dissimilarities(h))*qt*h.β
end

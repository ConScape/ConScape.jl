struct Habitat
    g::Grid
    cost::Cost
    C::SparseMatrixCSC{Float64,Int}
    Pref::SparseMatrixCSC{Float64,Int}
    # landmarks::Vector{Int}
end

"""
    Habitat(g::Grid, cost::Cost) -> Habitat

Construct a Habitat from a `g::Grid` based on a `cost::Cost` type.
"""
Habitat(g::Grid,
        cost::Cost) =
            Habitat(g,
                    cost,
                    mapnz(cost, g.A),
                    _Pref(g.A))

_Pref(A::SparseMatrixCSC) = Diagonal(inv.(vec(sum(A, dims=2)))) * A

function _W(Pref::SparseMatrixCSC, β::Real, C::SparseMatrixCSC)

    n = LinearAlgebra.checksquare(Pref)
    if LinearAlgebra.checksquare(C) != n
        throw(DimensionMismatch("Pref and C must have same size"))
    end

    return Pref .* exp.((-).(β) .* C)
end

_W(h::Habitat; β=nothing) = _W(h.Pref, β, h.C)

"""
    RSP_full_betweenness_qweighted(h::Habitat; β=nothing) -> Matrix

Compute full RSP betweenness of all nodes weighted by source and target qualities.
"""
function RSP_full_betweenness_qweighted(h::Habitat; β=nothing)
    betvec = RSP_full_betweenness_qweighted(
        inv(Matrix(I - _W(h, β=β))),
        h.g.source_qualities[h.g.id_to_grid_coordinate_list],
        h.g.target_qualities[h.g.id_to_grid_coordinate_list])

    bet = zeros(h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end


"""
    RSP_full_betweenness_kweighted(h::Habitat; β=nothing) -> Matrix

Compute full RSP betweenness of all nodes weighted with proximity.
"""
function RSP_full_betweenness_kweighted(h::Habitat; β=nothing)
    W = _W(h, β=β)
    Z = inv(Matrix(I - W))
    similarities = map(t -> iszero(t) ? t : inv(h.cost)(t), RSP_dissimilarities(W, h.C, Z))
    betvec = RSP_full_betweenness_kweighted(Z,
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
    RSP_dissimilarities(h::Habitat; β=nothing) -> Matrix

Compute RSP expected costs or RSP dissimilarities from all nodes.
"""
RSP_dissimilarities(h::Habitat; β=nothing) = RSP_dissimilarities(_W(h, β=β), h.C)

RSP_free_energy_distance(h::Habitat; β=nothing) = RSP_free_energy_distance(inv(Matrix(I - _W(h, β=β))), β)

"""
    mean_kl_distance(h::Habitat; β=nothing) -> Real

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP dissimilarities for `h::Habitat` at the temperature `β`.
"""
function mean_kl_distance(h::Habitat; β=nothing)
    W = _W(h, β=β)
    Z = inv(Matrix(I - W))
    qs = h.g.source_qualities[h.g.id_to_grid_coordinate_list]
    qt = h.g.target_qualities[h.g.id_to_grid_coordinate_list]
    return qs'*(RSP_free_energy_distance(Z, β) - RSP_dissimilarities(W, h.C, Z))*qt*β
end

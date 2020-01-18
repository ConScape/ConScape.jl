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

function Habitat(g::Grid;
                 cost::Cost=MinusLog(),
                 β=nothing,
                 C::SparseMatrixCSC{Float64,Int}=mapnz(cost, g.A))

    Pref = _Pref(g.A)
    W    = _W(Pref, β, C)
    @debug("Computing fundamental matrix of non-absorbing paths (Z). Please be patient...")
    targetidx, targetnodes = _targetidx_and_nodes(g)
    Z    = (I - W)\Matrix(sparse(targetnodes,
                                 1:length(targetnodes),
                                 1.0,
                                 size(C, 1),
                                 length(targetnodes)))
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

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

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
    RSP_betweenness_qweighted(h::Habitat)::Matrix{Float64}

Compute full RSP betweenness of all edges weighted by source and target qualities. Returns a
sparse matrix where element (i,j) is the betweenness of edge (i,j).
"""
function RSP_edge_betweenness_qweighted(h::Habitat)

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    betmatrix = RSP_edge_betweenness_qweighted(
        h.W,
        h.Z,
        [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
        [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
        targetnodes)

    return betmatrix
end



"""
    RSP_betweenness_kweighted(h::Habitat)::Matrix{Float64}

Compute full RSP betweenness of all edges weighted by source and target qualities. Returns a
sparse matrix where element (i,j) is the betweenness of edge (i,j).
"""
function RSP_edge_betweenness_kweighted(h::Habitat; invcost=inv(h.cost), diagvalue=nothing)

    similarities = map(invcost, RSP_dissimilarities(h))
    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            similarities[i, j] = diagvalue
        end
    end

    betmatrix = RSP_edge_betweenness_kweighted(
        h.W,
        h.Z,
        [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
        [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
        similarities,
        targetnodes)

    return betmatrix
end



"""
    RSP_betweenness_kweighted(h::Habitat; [invcost=inv(h.cost)])::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.
"""
function RSP_betweenness_kweighted(h::Habitat; invcost=inv(h.cost), diagvalue=nothing)

    similarities = map(invcost, RSP_dissimilarities(h))

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            similarities[i, j] = diagvalue
        end
    end

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
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    return RSP_dissimilarities(h.W, h.C, h.Z, targetnodes)
end

RSP_free_energy_distance(h::Habitat) = RSP_free_energy_distance(h.Z, h.β)

"""
    mean_kl_divergence(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP dissimilarities for `h::Habitat`.
"""
function mean_kl_divergence(h::Habitat)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    qs = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qt = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx]
    return qs'*(RSP_free_energy_distance(h.Z, h.β, targetnodes) - RSP_dissimilarities(h))*qt*h.β
end


"""
    mean_lc_kl_divergence(h::Habitat)::Float64

Compute the mean Kullback–Leibler divergence between the least-cost path and the random path distribution for `h::Habitat`, weighted by the qualities of the source and target node.
"""
function mean_lc_kl_divergence(h::Habitat)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    div = hcat([least_cost_kl_divergence(h.C, h.Pref, i) for i in targetnodes]...)
    qs = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qt = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx]
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
    RSP_functionality(h::Habitat; [invcost=inv(h.cost), diagvalue=nothing])::Matrix{Float64}

Compute RSP functionality of all nodes. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of similarities, i.e. after applying the inverse cost function to the matrix of dissimilarities.
When nothing is specified, the diagonal elements won't be adjusted.
"""
function RSP_functionality(h::Habitat; invcost=inv(h.cost), diagvalue=nothing)

    S = RSP_dissimilarities(h)
    map!(invcost, S, S)

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            S[i, j] = diagvalue
        end
    end

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in targetidx]

    funvec = RSP_functionality(qˢ, qᵗ, S)

    func = sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                  [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                  funvec,
                  h.g.nrows,
                  h.g.ncols)

    return func
end

function RSP_functionality(h::Habitat,
                           cell::CartesianIndex{2};
                           invcost=inv(h.cost),
                           diagvalue=nothing,
                           avalue=floatmin(), # smallest non-zero value
                           qˢvalue=0.0,
                           qᵗvalue=0.0)

    if avalue <= 0.0
        throw("Affinity value has to be positive. Otherwise the graph will become disconnected.")
    end

    # Compute (linear) node indices from (cartesian) grid indices
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    node = findfirst(isequal(cell), h.g.id_to_grid_coordinate_list)

    # Check that cell is in targetidx
    if cell ∉ targetidx
        throw(ArgumentError("Computing adjusted functionality is only supported for target cells"))
    end

    newA = copy(h.g.A)
    newA[:, node] .= ifelse.(iszero.(newA[:, node]), 0, avalue)
    newA[node, :] .= ifelse.(iszero.(newA[node, :]), 0, avalue)

    newsource_qualities = copy(h.g.source_qualities)
    newsource_qualities[cell] = qˢvalue
    newtarget_qualities = copy(h.g.target_qualities)
    newtarget_qualities[cell] = qᵗvalue

    newg = Grid(h.g.nrows,
                h.g.ncols,
                newA,
                h.g.id_to_grid_coordinate_list,
                newsource_qualities,
                newtarget_qualities)

    newh = Habitat(newg, β=h.β, cost=h.cost)

    return RSP_functionality(newh)
end

"""
    RSP_criticality(h::Habitat[;
                    invcost=inv(h.cost),
                    diagvalue=nothing,
                    avalue=floatmin(),
                    qˢvalue=0.0,
                    qᵗvalue=0.0])

Compute the landscape criticality for each target cell by setting setting affinities
for the cell to `avalue` as well as the source and target qualities associated with
the cell to `qˢvalue` and `qᵗvalue` respectively. It is required that `avalue` is
positive to avoid that the graph becomes disconnected. See `RSP_functionality`(@ref)
for the remaining arguments.
"""
function RSP_criticality(h::Habitat;
                         invcost=inv(h.cost),
                         diagvalue=nothing,
                         avalue=floatmin(),
                         qˢvalue=0.0,
                         qᵗvalue=0.0)

    targetidx = CartesianIndex.(findnz(h.g.target_qualities)[1:2]...)
    nl = length(targetidx)
    reference_functionality = sum(RSP_functionality(h, invcost=invcost, diagvalue=diagvalue))
    critvec = fill(reference_functionality, nl)

    @showprogress 1 "Computing criticality..." for i in 1:nl
        critvec[i] -= sum(RSP_functionality(
                            h,
                            targetidx[i];
                            invcost=invcost,
                            diagvalue=diagvalue,
                            avalue=avalue,
                            qˢvalue=qˢvalue,
                            qᵗvalue=qᵗvalue))
    end

    return SparseMatrixCSC(h.g.target_qualities.m,
                           h.g.target_qualities.n,
                           copy(h.g.target_qualities.colptr),
                           copy(h.g.target_qualities.rowval),
                           critvec)
end

# using Base.Threads

function LF_sensitivity(h::Habitat; invcost=inv(h.cost), exp_prox_scaling=1.)

    if h.cost == Inv() && exp_prox_scaling !== 1.
        throw(ArgumentError("exp_prox_scaling can be other than 1 only when using exponential proximity transformation"))
    end

    # Derivative of costs w.r.t. affinities:
    # TODO: Implement these as properties of Costs:
    K = map(invcost, RSP_dissimilarities(h)./exp_prox_scaling)
    n = length(h.g.id_to_grid_coordinate_list)
    K[1:n+1:end] .= 1
    K[isinf.(K)] .= 0

    if h.cost == MinusLog()
        diff_C_A = -mapnz(inv, h.g.A)
        diff_K_D = -K./exp_prox_scaling
    elseif h.cost == Inv()
        diff_C_A = -mapnz(x -> inv(x^2), h.g.A)
        diff_K_D = -K.^2
    end

    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list]

    landmarks = 1:length(h.g.id_to_grid_coordinate_list) # TODO: Include consideration of landmarks

    S_e_total, S_e_aff, S_e_cost = LF_sensitivity(qˢ, qᵗ, h.g.A, h.C, h.β, diff_C_A, diff_K_D, h.W, h.Z, invcost, landmarks)

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))


    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            h.g.nrows,
                                            h.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end








function LF_power_mean_sensitivity(h::Habitat; invcost=inv(h.cost))
    # Now assumes h.cost = MinusLog
    # TODO: Implement the derivatives of a2c and d2k transformations

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list]

    if h.cost == MinusLog()
        diff_C_A = -mapnz(inv, h.g.A)
    elseif h.cost == Inv()
        diff_C_A = mapnz(x -> x^2, h.g.A)
        diff_C_A = -mapnz(inv, diff_C_A)
    end
    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    landmarks = 1:length(h.g.id_to_grid_coordinate_list) # TODO: Include consideration of landmarks
    S_e_total, S_e_aff, S_e_cost = LF_power_mean_sensitivity(qˢ, qᵗ, h.g.A, h.β, diff_C_A, h.W, h.Z, landmarks)

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))

    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            h.g.nrows,
                                            h.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end

function LF_sensitivity_simulation(h::Habitat)

    LF_orig = ConScape.RSP_functionality(h)
    g = h.g

    epsi = 1e-8

    edge_sensitivities = copy(g.A)

    n = length(g.id_to_grid_coordinate_list)
    for i in 1:n
        Succ_i = findall(g.A[i,:].>0)

        for j in Succ_i
            gnew = deepcopy(g)
            gnew.A[i,j] += epsi
            # gnew.A[i,j] *= (1+epsi)

            hnew = ConScape.Habitat(gnew, β=h.β)
            LF_new = ConScape.RSP_functionality(hnew)
            edge_sensitivities[i,j] = sum(LF_new-LF_orig)/epsi # (gnew.A[i,j]-g.A[i,j])
        end
    end

    node_sensitivities = vec(sum(edge_sensitivities, dims=1))


    return Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                  [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                  node_sensitivities,
                  h.g.nrows,
                  h.g.ncols)),
           edge_sensitivities

end



function LF_power_mean_sensitivity_simulation(h::Habitat)

    g = h.g

    qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
    qᵗ = [g.target_qualities[i] for i in g.id_to_grid_coordinate_list]

    K = copy(h.Z)
    K ./= diag(h.Z)'
    K .^= inv(h.β) # \mathcal{Z}^{1/β}
    LF_orig = ConScape.RSP_functionality(qˢ, qᵗ, K)

    epsi = 1e-8

    edge_sensitivities = copy(g.A)

    n = length(g.id_to_grid_coordinate_list)
    @showprogress for i in 1:n
        Succ_i = findall(g.A[i,:].>0)

        for j in Succ_i
            gnew = deepcopy(g)
            gnew.A[i,j] += epsi
            # gnew.A[i,j] *= (1+epsi)

            hnew = ConScape.Habitat(gnew, β=h.β)

            Snew = copy(hnew.Z)
            Snew ./= diag(hnew.Z)'
            Snew .^= inv(hnew.β) # \mathcal{Z}^{1/β}
            LF_new = ConScape.RSP_functionality(qˢ, qᵗ, Snew)

            edge_sensitivities[i,j] = sum(LF_new-LF_orig)/epsi # (gnew.A[i,j]-g.A[i,j])
        end
    end

    node_sensitivities = vec(sum(edge_sensitivities, dims=1))


    return Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                  [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                  node_sensitivities,
                  g.nrows,
                  g.ncols)),
           edge_sensitivities

end

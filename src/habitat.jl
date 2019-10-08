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
    @debug("Compututing fundamental matrix of non-absorbing paths (Z). Please be patient...")
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
    RSP_betweenness_kweighted(h::Habitat; [invcost=inv(h.cost)])::Matrix{Float64}

Compute full RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
dissimilarities to convert it to a matrix of similarities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the dissimilarities.
"""
function RSP_betweenness_kweighted(h::Habitat; invcost=inv(h.cost))

    similarities = map(invcost, RSP_dissimilarities(h))

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

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

function LF_sensitivity(h::Habitat)

end

function LF_sensitivity_simulation(h::Habitat, invcost=inv(h.cost))

    expC_orig = RSP_dissimilarities(h)
    K_orig = map(invcost,expC_orig);

    g = h.g

    epsi = 1e-5

    edge_sensitivities = copy(g.A)

    n = length(g.id_to_grid_coordinate_list)
    for i in 1:n
        Succ_i = findall(g.A[i,:].>0)
        K_orig_i = copy(K_orig)
        K_orig_i[:,i] .= 0

        for j in Succ_i
            gnew = deepcopy(g)
            gnew.A[i,j] += epsi

            hnew = ConScape.Habitat(gnew, β=h.β)
            expC_new = RSP_dissimilarities(hnew)

            Knew = map(invcost, expC_new)
            Knew[:,i] .= 0

            edge_sensitivities[i,j] = sum(sum(Knew - K_orig_i))/epsi
        end
    end

    node_sensitivities = vec(sum(edge_sensitivities, dims=1))


    return sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                  [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                  node_sensitivities,
                  h.g.nrows,
                  h.g.ncols)

end

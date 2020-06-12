abstract type Cost end
struct MinusLog     <: Cost end
struct ExpMinus     <: Cost end
struct Inv          <: Cost end
struct OddsAgainst  <: Cost end
struct OddsFor      <: Cost end

(::MinusLog)(x::Number)     = -log(x)
(::ExpMinus)(x::Number)     = exp(-x)
(::Inv)(x::Number)          = inv(x)
(::OddsAgainst)(x::Number)  = inv(x) - 1
(::OddsFor)(x::Number)      = x/(1 - x)


Base.inv(::MinusLog)     = ExpMinus()
Base.inv(::ExpMinus)     = MinusLog()
Base.inv(::Inv)          = Inv()
Base.inv(::OddsAgainst)  = OddsFor()
Base.inv(::OddsFor)      = OddsAgainst()

abstract type ConnectivityFunction <: Function end
abstract type DistanceFunction <: ConnectivityFunction end
abstract type ProximityFunction <: ConnectivityFunction end

struct expected_cost         <: DistanceFunction end
struct free_energy_distance  <: DistanceFunction end

struct survival_probability  <: ProximityFunction end
struct power_mean_proximity  <: ProximityFunction end

struct GridRSP
    g::Grid
    cost::Cost
    β::Float64
    C::SparseMatrixCSC{Float64,Int}
    Pref::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    Z::Matrix{Float64}
end

"""
    GridRSP(g::Grid; cost::Cost=MinusLog(), β=nothing)::GridRSP

Construct a GridRSP from a `g::Grid` based on a `cost::Cost` type and the temperature `β::Real`.
"""

function GridRSP(g::Grid;
                 cost::Cost=MinusLog(),
                 β=nothing,
                 C::SparseMatrixCSC{Float64,Int}=mapnz(cost, g.affinities))

    if any(t -> t < 0, nonzeros(C))
        throw(ArgumentError("The cost matrix C can have only non-negative elements. Perhaps you should change the cost function?"))
    end

    Pref = _Pref(g.affinities)
    W    = _W(Pref, β, C)

    @debug("Computing fundamental matrix of non-absorbing paths (Z). Please be patient...")
    targetidx, targetnodes = _targetidx_and_nodes(g)
    Z    = (I - W)\Matrix(sparse(targetnodes,
                                 1:length(targetnodes),
                                 1.0,
                                 size(C, 1),
                                 length(targetnodes)))
    # Check that values in Z are not too small:
    if minimum(Z)*minimum(nonzeros(C.*W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing β."
    end

    return GridRSP(g, cost, β, C, Pref, W, Z)
end

function Base.show(io::IO, ::MIME"text/plain", h::GridRSP)
    print(io, summary(h), " of size ", h.g.nrows, "x", h.g.ncols)
end

function Base.show(io::IO, ::MIME"text/html", h::GridRSP)
    t = string(summary(h), " of size ", h.g.nrows, "x", h.g.ncols)
    write(io, "<h4>$t</h4>")
    show(io, MIME"text/html"(), plot_outdegrees(h.g))
end

"""
    betweenness_qweighted(h::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all nodes weighted by source and target qualities.
"""
function betweenness_qweighted(h::GridRSP)

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
    edge_betweenness_qweighted(h::GridRSP)::Matrix{Float64}

Compute RSP betweenness of all edges weighted by source and target qualities. Returns a
sparse matrix where element (i,j) is the betweenness of edge (i,j).
"""
function edge_betweenness_qweighted(h::GridRSP)

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
    betweenness_kweighted(h::GridRSP; [invcost=inv(h.cost), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

Compute RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of distances.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.
"""
function betweenness_kweighted(h::GridRSP; invcost=inv(h.cost), diagvalue=nothing)

    proximities = map(invcost, expected_cost(h))

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            proximities[i, j] = diagvalue
        end
    end

    betvec = RSP_betweenness_kweighted(
        h.W,
        h.Z,
        [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
        [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
        proximities,
        targetnodes)

    bet = fill(NaN, h.g.nrows, h.g.ncols)
    for (i, v) in enumerate(betvec)
        bet[h.g.id_to_grid_coordinate_list[i]] = v
    end

    return bet
end



"""
    edge_betweenness_kweighted(h::GridRSP; [invcost=inv(h.cost), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

    Compute RSP betweenness of all edges weighted by qualities of source s and target t and the proximity between s and t. Returns a
    sparse matrix where element (i,j) is the betweenness of edge (i,j).

    The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
    of proximities, i.e. after applying the inverse cost function to the matrix of expected costs.
    When nothing is specified, the diagonal elements won't be adjusted.
"""
function edge_betweenness_kweighted(h::GridRSP; invcost=inv(h.cost), diagvalue=nothing)

    proximities = map(invcost, expected_cost(h))
    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            proximities[i, j] = diagvalue
        end
    end


    betmatrix = RSP_edge_betweenness_kweighted(
        h.W,
        h.Z,
        [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list],
        [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx],
        proximities,
        targetnodes)

    return betmatrix
end



"""
    expected_cost(h::GridRSP)::Matrix{Float64}

Compute RSP expected costs from all nodes.
"""
function expected_cost(h::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    return RSP_expected_cost(h.W, h.C, h.Z, targetnodes)
end

function free_energy_distance(h::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    return RSP_free_energy_distance(h.Z, h.β, targetnodes)
end

function survival_probability(h::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    return RSP_survival_probability(h.Z, h.β, targetnodes)
end

function power_mean_proximity(h::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    return RSP_power_mean_proximity(h.Z, h.β, targetnodes)
end

"""
    mean_kl_divergence(h::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the free energy distances and the RSP expected costs for `h::GridRSP`.
"""
function mean_kl_divergence(h::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(h.g)
    qs = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qt = [h.g.target_qualities[i] for i in h.g.id_to_grid_coordinate_list ∩ targetidx]
    return qs'*(RSP_free_energy_distance(h.Z, h.β, targetnodes) - expected_cost(h))*qt*h.β
end


"""
    mean_lc_kl_divergence(h::GridRSP)::Float64

Compute the mean Kullback–Leibler divergence between the least-cost path and the random path distribution for `h::GridRSP`, weighted by the qualities of the source and target node.
"""
function mean_lc_kl_divergence(h::GridRSP)
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
    least_cost_kl_divergence(h::GridRSP, target::Tuple{Int,Int})

Compute the least cost Kullback-Leibler divergence from each cell in the g in
`h` to the `target` cell.
"""
function least_cost_kl_divergence(h::GridRSP, target::Tuple{Int,Int})

    targetnode = findfirst(isequal(CartesianIndex(target)), h.g.id_to_grid_coordinate_list)
    if targetnode === nothing
        throw(ArgumentError("target cell not found"))
    end

    div = least_cost_kl_divergence(h.C, h.Pref, targetnode)

    return reshape(div, h.g.nrows, h.g.ncols)
end

"""
    functionality(h::GridRSP; [connectivity_function=expected_cost, invcost=inv(h.cost), diagvalue=nothing])::Matrix{Float64}

Compute RSP functionality of all nodes. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the proximities.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

`connectivity_function` determines which function is used for computing the matrix of proximities.
If `connectivity_function` is a `DistanceFunction`, then it is used for computing distances, which
is converted to proximities using `invcost`. If `connectivity_function` is a `ProximityFunction`,
then proximities are computed directly using it. The default is `expected_cost`.
"""
function functionality(h::GridRSP;
                       connectivity_function=expected_cost,
                       invcost=inv(h.cost),
                       diagvalue=nothing)

    S = connectivity_function(h)

    if connectivity_function <: DistanceFunction
        map!(invcost, S, S)
    end

    return functionality(h, S, diagvalue=diagvalue)
end

function functionality(h::GridRSP, S::Matrix; diagvalue::Union{Nothing,Real}=nothing)

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

function functionality(h::GridRSP,
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

    newA = copy(h.g.affinities)
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

    newh = GridRSP(newg, β=h.β, cost=h.cost)

    return functionality(newh)
end

"""
    criticality(h::GridRSP[;
                    invcost=inv(h.cost),
                    diagvalue=nothing,
                    avalue=floatmin(),
                    qˢvalue=0.0,
                    qᵗvalue=0.0])

Compute the landscape criticality for each target cell by setting setting affinities
for the cell to `avalue` as well as the source and target qualities associated with
the cell to `qˢvalue` and `qᵗvalue` respectively. It is required that `avalue` is
positive to avoid that the graph becomes disconnected. See `functionality`(@ref)
for the remaining arguments.
"""
function criticality(h::GridRSP;
                     invcost=inv(h.cost),
                     diagvalue=nothing,
                     avalue=floatmin(),
                     qˢvalue=0.0,
                     qᵗvalue=0.0)

    targetidx = CartesianIndex.(findnz(h.g.target_qualities)[1:2]...)
    nl = length(targetidx)
    reference_functionality = sum(functionality(h, invcost=invcost, diagvalue=diagvalue))
    critvec = fill(reference_functionality, nl)

    @showprogress 1 "Computing criticality..." for i in 1:nl
        critvec[i] -= sum(functionality(
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


"""
    LF_sensitivity(h::GridRSP; invcost=inv(h.cost), exp_prox_scaling::Real=1., unitless::Bool=true)::Matrix{Float64}

Compute the sensitivity of Landscape Functionality with respect to perturbation of
affinities on incoming edges of a node. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of the distances.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.

 - `exp_prox_scaling`: the scaling parameter of the exponential cost function.
 - `unitless`: A boolean deciding whether the output is the "unitless" derivative, i.e., ``\\frac{\\mathrm{d} f}{\\mathrm{d} \\log x}``, or the standard derivative

"""

# using Base.Threads

function LF_sensitivity(h::GridRSP; invcost=inv(h.cost),
                                    exp_prox_scaling::Real=1.,
                                    unitless::Bool=true,
                                    diagvalue=nothing)

    if h.cost == Inv() && exp_prox_scaling !== 1.
        throw(ArgumentError("exp_prox_scaling can be other than 1 only when using exponential proximity transformation"))
    end

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    # Derivative of costs w.r.t. affinities:
    # TODO: Implement these as properties of Costs:
    K = map(invcost, expected_cost(h) ./ exp_prox_scaling)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            K[i, j] = diagvalue
        end
    end

    K[isinf.(K)] .= 0.

    if h.cost == MinusLog()
        diff_C_A = -mapnz(inv, h.g.affinities)
        diff_K_D = -K./exp_prox_scaling
    elseif h.cost == Inv()
        diff_C_A = -mapnz(x -> inv(x^2), h.g.affinities)
        diff_K_D = -K.^2
    end

    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in targetidx]

    S_e_aff, S_e_cost = LF_sensitivity(h.g.affinities, h.C, h.β, h.W, h.Z, diff_K_D, qˢ, qᵗ, targetnodes)

    if unitless
        S_e_aff = S_e_aff.*h.g.affinities
        S_e_cost = S_e_cost.*h.g.affinities
    end

    S_e_total = S_e_aff .+ S_e_cost.*diff_C_A

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))


    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            h.g.nrows,
                                            h.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end


function LF_power_mean_sensitivity(h::GridRSP; invcost=inv(h.cost))
    # Now assumes h.cost = MinusLog

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in targetidx]

    if h.cost == MinusLog()
        diff_C_A = -mapnz(inv, h.g.affinities)
    elseif h.cost == Inv()
        diff_C_A = mapnz(x -> x^2, h.g.affinities)
        diff_C_A = -mapnz(inv, diff_C_A)
    end
    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    S_e_total, S_e_aff, S_e_cost = LF_power_mean_sensitivity(qˢ, qᵗ, h.g.affinities, h.β, diff_C_A, h.W, h.Z, targetnodes)

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))

    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in h.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in h.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            h.g.nrows,
                                            h.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end

function LF_sensitivity_simulation(h::GridRSP;
    invcost=inv(h.cost),
    exp_prox_scaling::Real=1.,
    unitless::Bool=true,
    diagvalue=nothing)

    LF_orig = functionality(h, diagvalue=diagvalue)
    g = h.g

    epsi = 1e-6

    edge_sensitivities = copy(g.affinities)

    n = length(g.id_to_grid_coordinate_list)
    @showprogress for i in 1:n
        Succ_i = findall(g.affinities[i,:].>0)

        for j in Succ_i
            gnew = deepcopy(g)
            gnew.affinities[i,j] += epsi
            # gnew.affinities[i,j] *= (1+epsi)

            hnew = GridRSP(gnew, β=h.β, cost=h.cost)
            LF_new = functionality(hnew, diagvalue=diagvalue)
            edge_sensitivities[i,j] = sum(LF_new-LF_orig)/epsi # (gnew.affinities[i,j]-g.affinities[i,j])
            if unitless
                edge_sensitivities[i,j] *= g.affinities[i,j]
            end
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



function LF_power_mean_sensitivity_simulation(h::GridRSP)

    g = h.g

    targetidx, targetnodes = _targetidx_and_nodes(h.g)

    qˢ = [h.g.source_qualities[i] for i in h.g.id_to_grid_coordinate_list]
    qᵗ = [h.g.target_qualities[i] for i in targetidx]

    K = copy(h.Z)
    K ./= [h.Z[targetnodes[i],i] for i in 1:length(targetnodes)]'
    K .^= inv(h.β) # \mathcal{Z}^{1/β}
    LF_orig = sum(RSP_functionality(qˢ, qᵗ, K))

    epsi = 1e-6

    edge_sensitivities = copy(g.affinities)

    n = length(g.id_to_grid_coordinate_list)
    @showprogress for i in 1:n
        Succ_i = findall(g.affinities[i,:].>0)

        for j in Succ_i
            gnew = deepcopy(g)
            gnew.affinities[i,j] += epsi
            # gnew.affinities[i,j] *= (1+epsi)

            hnew = GridRSP(gnew, β=h.β, cost=h.cost)

            Knew = copy(hnew.Z)
            Knew ./= [hnew.Z[targetnodes[i],i] for i in 1:length(targetnodes)]'
            Knew .^= inv(hnew.β) # \mathcal{Z}^{1/β}
            LF_new = sum(RSP_functionality(qˢ, qᵗ, Knew))

            edge_sensitivities[i,j] = (LF_new-LF_orig)/epsi # (gnew.affinities[i,j]-g.affinities[i,j])
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


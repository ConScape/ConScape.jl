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

Construct a GridRSP from a `g::Grid` based on a `cost::Cost` type and the temperature `β::Real`.
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
    betweenness_kweighted(grsp::GridRSP; [invcost=inv(grsp.g.costfunction), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

Compute RSP betweenness of all nodes weighted with proximity. Optionally, an inverse
cost function can be passed. The function will be applied elementwise to the matrix of
distances to convert it to a matrix of proximities. If no inverse cost function is
passed the the inverse of the cost function is used for the conversion of distances.

The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
of proximities, i.e. after applying the inverse cost function to the matrix of distances.
When nothing is specified, the diagonal elements won't be adjusted.
"""
function betweenness_kweighted(grsp::GridRSP; invcost=nothing, diagvalue=nothing)

    # Check that invcost function has been passed if no cost function is saved
    if invcost === nothing
        if grsp.g.costfunction === nothing
            throw(ArgumentError("no invcost function supplied and cost matrix in Grid isn't based on a cost function."))
        else
            invcost = inv(grsp.g.costfunction)
        end
    end

    proximities = map(invcost, expected_cost(grsp))
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
    edge_betweenness_kweighted(grsp::GridRSP; [invcost=inv(grsp.g.costfunction), diagvalue=nothing])::SparseMatrixCSC{Float64,Int}

    Compute RSP betweenness of all edges weighted by qualities of source s and target t and the proximity between s and t. Returns a
    sparse matrix where element (i,j) is the betweenness of edge (i,j).

    The optional `diagvalue` element specifies which value to use for the diagonal of the matrix
    of proximities, i.e. after applying the inverse cost function to the matrix of expected costs.
    When nothing is specified, the diagonal elements won't be adjusted.
"""
function edge_betweenness_kweighted(grsp::GridRSP; invcost=inv(grsp.g.costfunction), diagvalue=nothing)

    proximities = map(invcost, expected_cost(grsp))
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
    connected_habitat(grsp::GridRSP; [connectivity_function=expected_cost,
        invcost=inv(grsp.g.costfunction), diagvalue=nothing])::Matrix{Float64}

Compute RSP connected_habitat of all nodes. Optionally, an inverse
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
function connected_habitat(grsp::GridRSP;
                           connectivity_function=expected_cost,
                           invcost=nothing,
                           diagvalue=nothing)

    # Check that invcost function has been passed if no cost function is saved
    if invcost === nothing
        if grsp.g.costfunction === nothing
            throw(ArgumentError("no invcost function supplied and cost matrix in Grid isn't based on a cost function."))
        else
            invcost = inv(grsp.g.costfunction)
        end
    end

    S = connectivity_function(grsp)

    if connectivity_function <: DistanceFunction
        map!(invcost, S, S)
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
                           invcost=nothing,
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

    return connected_habitat(newh; diagvalue=diagvalue, invcost=invcost)
end

"""
    criticality(grsp::GridRSP[;
                invcost=inv(grsp.g.costfunction),
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
                     invcost=nothing,
                     diagvalue=nothing,
                     avalue=floatmin(),
                     qˢvalue=0.0,
                     qᵗvalue=0.0)

    targetidx = CartesianIndex.(findnz(grsp.g.target_qualities)[1:2]...)
    nl = length(targetidx)
    reference_connected_habitat = sum(connected_habitat(grsp, invcost=invcost, diagvalue=diagvalue))
    critvec = fill(reference_connected_habitat, nl)

    @showprogress 1 "Computing criticality..." for i in 1:nl
        critvec[i] -= sum(connected_habitat(
            grsp,
            targetidx[i];
            invcost=invcost,
            diagvalue=diagvalue,
            avalue=avalue,
            qˢvalue=qˢvalue,
            qᵗvalue=qᵗvalue))
    end

    return SparseMatrixCSC(grsp.g.target_qualities.m,
                           grsp.g.target_qualities.n,
                           copy(grsp.g.target_qualities.colptr),
                           copy(grsp.g.target_qualities.rowval),
                           critvec)
end


"""
    sensitivity(grsp::GridRSP; invcost=inv(grsp.g.costfunction), exp_prox_scaling::Real=1., unitless::Bool=true)::Matrix{Float64}

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
function sensitivity(grsp::GridRSP;
    invcost=nothing,
    exp_prox_scaling::Real=1.,
    unitless::Bool=true,
    diagvalue=nothing)

    if grsp.g.costfunction === nothing
        throw(ArgumentError("sensitivities are only defined when costs are functions of affinities"))
    end

    # Check that invcost function has been passed. If not then default to inverse of cost function
    if invcost === nothing
        invcost = inv(grsp.g.costfunction)
    end

    if grsp.g.costfunction == Inv() && exp_prox_scaling !== 1.
        throw(ArgumentError("exp_prox_scaling can different from 1 only when using exponential proximity transformation"))
    end

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    # Derivative of costs w.r.t. affinities:
    # TODO: Implement these as properties of Costs:
    K = map(invcost, expected_cost(grsp) ./ exp_prox_scaling)

    if diagvalue !== nothing
        for (j, i) in enumerate(targetnodes)
            K[i, j] = diagvalue
        end
    end

    K[isinf.(K)] .= 0.

    if grsp.g.costfunction == MinusLog()
        diff_C_A = -mapnz(inv, grsp.g.affinities)
        diff_K_D = -K./exp_prox_scaling
    elseif grsp.g.costfunction == Inv()
        diff_C_A = -mapnz(x -> inv(x^2), grsp.g.affinities)
        diff_K_D = -K.^2
    end

    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    qˢ = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qᵗ = [grsp.g.target_qualities[i] for i in targetidx]

    S_e_aff, S_e_cost = LF_sensitivity(
        grsp.g.affinities,
        grsp.g.costmatrix,
        grsp.β,
        grsp.W,
        grsp.Z,
        diff_K_D,
        qˢ,
        qᵗ,
        targetnodes)

    if unitless
        S_e_aff = S_e_aff.*grsp.g.affinities
        S_e_cost = S_e_cost.*grsp.g.affinities
    end

    S_e_total = S_e_aff .+ S_e_cost.*diff_C_A

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))


    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in grsp.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            grsp.g.nrows,
                                            grsp.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end


function power_mean_sensitivity(grsp::GridRSP; invcost=inv(grsp.g.costfunction))
    # Now assumes grsp.g.costfunction = MinusLog

    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)

    qˢ = [grsp.g.source_qualities[i] for i in grsp.g.id_to_grid_coordinate_list]
    qᵗ = [grsp.g.target_qualities[i] for i in targetidx]

    if grsp.g.costfunction == MinusLog()
        diff_C_A = -mapnz(inv, grsp.g.affinities)
    elseif grsp.g.costfunction == Inv()
        diff_C_A = mapnz(x -> x^2, grsp.g.affinities)
        diff_C_A = -mapnz(inv, diff_C_A)
    end
    # diff_C_A[Idx] = -1./A[Idx]; # derivative when c_ij = -log(a_ij)
    # diff_C_A(Idx) = -1./(A(Idx))^2; # derivative when c_ij = 1/a_ij

    S_e_total, S_e_aff, S_e_cost = LF_power_mean_sensitivity(qˢ, qᵗ, grsp.g.affinities, grsp.β, diff_C_A, grsp.W, grsp.Z, targetnodes)

    node_sensitivity_vec = vec(sum(S_e_total, dims=1))

    node_sensitivity_matrix = Matrix(sparse([ij[1] for ij in grsp.g.id_to_grid_coordinate_list],
                                            [ij[2] for ij in grsp.g.id_to_grid_coordinate_list],
                                            node_sensitivity_vec,
                                            grsp.g.nrows,
                                            grsp.g.ncols))

    return node_sensitivity_matrix, S_e_total, S_e_aff, S_e_cost

end

function sensitivity_simulation(grsp::GridRSP;
    exp_prox_scaling::Real=1.,
    unitless::Bool=true,
    diagvalue=nothing)

    lf = connected_habitat(grsp, diagvalue=diagvalue)
    g = grsp.g

    epsi = 1e-6

    edge_sensitivities = copy(g.affinities)

    n = length(g.id_to_grid_coordinate_list)
    @showprogress for i in 1:n
        Succ_i = findall(g.affinities[i,:].>0)

        for j in Succ_i
            new_affinities = copy(g.affinities)
            new_affinities[i, j] += epsi
            new_g = Grid(size(g)...,
                new_affinities,
                g.costfunction,
                mapnz(g.costfunction, new_affinities),
                g.id_to_grid_coordinate_list,
                g.source_qualities,
                g.target_qualities)

            new_grsp = GridRSP(new_g, β=grsp.β)
            new_lf = connected_habitat(new_grsp, diagvalue=diagvalue)

            edge_sensitivities[i,j] = sum(new_lf - lf)/epsi # (gnew.affinities[i,j]-g.affinities[i,j])
            if unitless
                edge_sensitivities[i,j] *= g.affinities[i,j]
            end
        end
    end

    node_sensitivities = vec(sum(edge_sensitivities, dims=1))

    return Matrix(sparse(
        [ij[1] for ij in g.id_to_grid_coordinate_list],
        [ij[2] for ij in g.id_to_grid_coordinate_list],
        node_sensitivities,
        g.nrows,
        g.ncols)), edge_sensitivities
end



function power_mean_sensitivity_simulation(grsp::GridRSP)

    g = grsp.g

    targetidx, targetnodes = _targetidx_and_nodes(g)

    qˢ = [g.source_qualities[i] for i in g.id_to_grid_coordinate_list]
    qᵗ = [g.target_qualities[i] for i in targetidx]

    K = copy(grsp.Z)
    K ./= [grsp.Z[targetnodes[i],i] for i in 1:length(targetnodes)]'
    K .^= inv(grsp.β) # \mathcal{Z}^{1/β}
    lf = sum(connected_habitat(qˢ, qᵗ, K))

    epsi = 1e-6

    edge_sensitivities = copy(g.affinities)

    n = length(g.id_to_grid_coordinate_list)
    @showprogress for i in 1:n
        Succ_i = findall(g.affinities[i,:].>0)

        for j in Succ_i
            new_affinities = copy(g.affinities)
            new_affinities[i, j] += epsi
            new_g = Grid(size(g)...,
                new_affinities,
                g.costfunction,
                mapnz(g.costfunction, new_affinities),
                g.id_to_grid_coordinate_list,
                g.source_qualities,
                g.target_qualities)

            new_grsp = GridRSP(new_g, β=grsp.β)

            new_K = copy(new_grsp.Z)
            new_K ./= [new_grsp.Z[targetnodes[i], i] for i in 1:length(targetnodes)]'
            new_K .^= inv(new_grsp.β) # \mathcal{Z}^{1/β}
            new_lf = sum(connected_habitat(qˢ, qᵗ, new_K))

            edge_sensitivities[i,j] = (new_lf - lf)/epsi # (gnew.affinities[i,j]-g.affinities[i,j])
        end
    end

    node_sensitivities = vec(sum(edge_sensitivities, dims=1))

    return Matrix(sparse(
        [ij[1] for ij in g.id_to_grid_coordinate_list],
        [ij[2] for ij in g.id_to_grid_coordinate_list],
        node_sensitivities,
        g.nrows,
        g.ncols)), edge_sensitivities

end


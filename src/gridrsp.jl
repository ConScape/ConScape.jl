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

abstract type Operation end 
abstract type RSPOperation <: Operation end

@kwdef struct ComputeAssesment{O,Z,L,M,F}
    op::O
    zmax::Z
    lumax::L
    totalmem::T
    flops::F
    # Something else?
end

"""
    allocate(co::ComputeAssesment)

Allocate memory required to run `compute` for the assessed ops.

The returned object can be passed as the `allocs` keyword to `compute`.
"""
function allocate(co::ComputeAssesment)
    zmax = co.zmax
    # But actually do this with GenericMemory using Julia v1.11
    Z = Matrix{Float64}(undef, co.zmax) 
    S = sparse(1:zmax[1], 1:zmax[2], 1.0, zmax...)
    L = lu(S)
    # Just return a NamedTuple for now
    return (; Z, S, L)
end

"""
    compute(o::Operation, grsp::GridRSP)

Compute operation `o` on precomputed grid `grsp`.
"""
function compute end

"""
    assess(o::Operation, g::Grid; grid_assessment)

Assess the memory and compute requirements of operation
`o` on grid `g`. This can be used to indicate memory
and time reequiremtents on a cluster
"""
function assess end

"""
    Operations(ops...; θ)

Combine multiple compute operations into a single object, 
to be run in the same job.
"""
@kwdef struct Operations{O,T,A} <: Operation
    op::O
    θ::T=nothing
end
Operations(op::O) where O<:Tuple = Operations{O}(op)
Operations(args...) = Operations(args)
Operations(::Operations; θ=nothing)= Operations(o.op; θ)

function compute(o::Operation, grsp::Grid)
    compute(o, GridRSP(g; allocs=o.allocs); θ=o.θ)
end
function compute(o::Operations, grsp::GridRSP) 
    map(compute, o.op)
    # Something else?
end

function assess(op::Operations, g::Grid; grid_assessment=asses(g)) 
    as = map(o -> asses(o, g), op.op)
    # some code to combine
end

"""
    WindowedOperations(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.


"""
@kwdef struct WindowedOperations{O,SS,WS,WC} <: Operation
    op::O
    sourcesize::SS
    size::WS
    centers::WC
end
function WindowedOperations(op::O, sourcesize::Tuple, windowsize::Tuple, windowcenters::AbstractMatrix; 
    θ::T=nothing
) where O<:Tuple 
    WindowedOperations{O}(Operation(op; θ), sourcesize, windowsize, windowcenters)
end

function compute(o::WindowedOperation, source::AbstractMatrix, target::AbstractMatrix)
    compute(o, GridRSP(g; allocs=o.allocs); θ=o.θ)
end

function assess(op::WindowedOperations, g::Grid) 
    window_assessments = map(_windows(op, g)) do w
        ca = assess(op.op, w)
    end
    maximums = reduce(window_assessments) do acc, a
        (; totalmem=max(acc.totalmem, a.totalmem),
           zmax=max(acc.zmax, a.zmax),
           lumax=max(acc.lumax, a.lumax),
        )
    end
    sums = reduce(window_assessments) do acc, a
        (; flops=acc.flops + a.flops)
    end
    ComputeAssesment(; op=op.op, maximums..., sums...)
end

@kwdef struct BetweennessQ <: RSPResult end

compute(r::BetweennessQ, grsp::GridRSP) = betweenness_qweighted(grsp)
assess(r::BetweennessQ, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct EdgeBetweenness <: RSPResult end

compute(r::EdgeBetweenness, grsp::GridRSP) = betweenness_qweighted(grsp)
assess(r::EdgeBetweenness, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct BetweennessK{CV,DT,DV} <: RSPResult 
    connectivity_function::CV=expected_cost
    distance_transformation::DT=nothing
    diagvalue::DV=nothing
end

compute(r::BetweennessK, grsp::GridRSP) = betweenness_kweighted(grsp; keywords(r)...)
assess(r::BetweennessK, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct EdgeBetweennessK{CV,DT,DV} <: RSPResult 
    distance_transformation::DT=inv(grsp.g.costfunction)
    diagvalue::DV=nothing
end

compute(r::EdgeBetweennessK, grsp::GridRSP) = edge_betweenness_kweighted(grsp; keywords(r)...)
assess(r::EdgeBetweennessK, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct ExpectedCost <: RSPResult end

compute(r::ExpectedCost, grsp::GridRSP) = expected_cost(grsp)
assess(r::ExpectedCost, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

"""
    expected_cost(grsp::GridRSP)::Matrix{Float64}

Compute RSP expected costs from all nodes.
"""
function expected_cost(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_expected_cost(grsp.W, grsp.g.costmatrix, grsp.Z, targetnodes)
end

@kwdef struct FreeDnergyDistance <: RSPResult end

compute(r::FreeDnergyDistance, grsp::GridRSP) = free_energy_distance(grsp)
assess(r::FreeDnergyDistance, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

function free_energy_distance(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_free_energy_distance(grsp.Z, grsp.θ, targetnodes)
end

@kwdef struct SurvivalProbability <: RSPResult end

compute(r::SurvivalProbability, grsp::GridRSP) = survival_probability(grsp)
assess(r::SurvivalProbability, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

function survival_probability(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_survival_probability(grsp.Z, grsp.θ, targetnodes)
end

@kwdef struct PowerMeanProximity <: RSPResult end

compute(r::PowerMeanProximity, grsp::GridRSP) = power_mean_proximity(grsp)
assess(r::PowerMeanProximity, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

function power_mean_proximity(grsp::GridRSP)
    targetidx, targetnodes = _targetidx_and_nodes(grsp.g)
    return RSP_power_mean_proximity(grsp.Z, grsp.θ, targetnodes)
end

@kwdef struct LeastCostDistance <: RSPResult end

compute(r::LeastCostDistance, grsp::GridRSP) = least_cost_distance(grsp)
assess(r::LeastCostDistance, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

least_cost_distance(grsp::GridRSP) = least_cost_distance(grsp.g)

@kwdef struct MeanKullbackLeiblerDivergence <: RSPResult end

compute(r::MeanKullbackLeiblerDivergence, grsp::GridRSP) = mean_kl_divergence(grsp)
assess(r::MeanKullbackLeiblerDivergence, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct MeanLeastCostKullbackLeiblerDivergence <: RSPResult end

compute(r::MeanLeastCostKullbackLeiblerDivergence, grsp::GridRSP) = mean_kl_divergence(grsp)
assess(r::MeanLeastCostKullbackLeiblerDivergence, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 


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

@kwdef struct LeastCostKullbackLeiblerDivergence <: RSPResult end

compute(r::LeastCostKullbackLeiblerDivergence, grsp::GridRSP) = least_cost_kl_divergence(grsp)
assess(r::LeastCostKullbackLeiblerDivergence, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct ConnectedHabitat{CV,DT,DV} <: RSPResult
    # TODO not sure which kw to use here
    connectivity_function::CV=expected_cost
    distance_transformation::DT=nothing
    diagvalue::DV=nothing
    θ::Union{Nothing,Real}=nothing
    approx::Bool=false
end

compute(r::ConnectedHabitat, grsp::GridRSP) = eigmax(grsp; keywords(r)...)
assess(r::ConnectedHabitat, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct EigMax{F,DT,DV,T} <: RSPResult 
    connectivity_function::F=expected_cost
    Tdistance_transformation::DT=nothing
    diagvalue::DV=nothing
    tol::T=1e-14
end

compute(r::EigMax, grsp::GridRSP) = eigmax(grsp; keywords(r)...)
assess(r::EigMax, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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

@kwdef struct Criticality{DT,DV,AV,QT,QS} <: RSPResult 
    distance_transformation::DT=inv(grsp.g.costfunction)
    diagvalue::DV=nothing
    avalue::AV=floatmin()
    qˢvalue::QS=0.0
    qᵗvalue::QT=0.0
end

compute(r::Criticality, grsp::GridRSP) = criticality(grsp; keywords(r)...)
assess(r::Criticality, grsp::Grid; grid_assessment=asses(g)) = nothing # TODO 

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


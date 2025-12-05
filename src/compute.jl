@inline function get_or_compute_target!(ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)
    haskey(st, x) && return st[x]
    val = compute_target(m, ti)
    if returntrait(m) isa ReturnSpatial
        st[x] = readonlyarray(val)
        return readonlyarray(val)
    else
        return val
    end
end
@inline function get_or_compute_target!(ti::TargetInit, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]

    val = target_precalculation!(ti, x)
    # TODO: storing other types
    if val isa Vector
        st[x] = readonlyarray(val)
    end
    return val
end

# Most measures dont need to deal with `update_connectedgraph_output` and just use `compute`
function compute_target!(output, l::Level, m::Measure, ti::TargetInit)
    val = compute_target(m, ti)
    update_connectedgraph_output!(output, l, m, ti, val)

    return output
end

#-------------------------------------------------------------------------------------------
# ConnectedGraph level measures

# Compute Target level measures for a connected subgraph
# function compute_target(m::Measure, cgi::ConnectedGraphInit)
#     output = allocate_output(
#     ConnectedGraphLevel(), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi)
#     )
#     for target in targetids(cgi)
#         ti = TargetInit(cgi, target)
#         compute_target!(output, m, ti)
#     end
#     return finalize_connectedgraph_output!(output, m, cgi)
# end

# Allocate sqauare matrix of target * target size
function allocate_output(
    l::GridGraphLevel,
    ::ReturnCustom,
    m::EigMax,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    T = Tuple{Vector{Float64},Array{Float64,0},Vector{Float64}}
    l => Vector{T}(undef, length(connectedgraphs))
end
function allocate_output(
    l::Union{ConnectedGraphLevel,GridGraphLevel}, # TargetLevel would be very expensive
    ::ReturnCustom,
    m::EigMax,
    ::ConScapeProblem,
    ::GridGraph,
    cg::ConnectedGraph,
    precalculation
)
    n = length(sourceids(cg))
    vʳ = fill(NaN, n)
    λ = fill(0.0)
    vˡ = zeros(n)
    return l => (vʳ, λ, vˡ)
end

function allocate_intermediate(::EigMax, cgi::ConnectedGraphInit)
    (; C, W, Z_full) = cgi
    m = length(targetids(cgi))
    targetnodes = map(x -> x.node, targetids(cgi))
    nontargetnodes = setdiff(1:m, targetnodes)
    Mtarget = zeros(m, m)
    Mnontarget = zeros(length(nontargetnodes), m)
    return (; Mtarget, Mnontarget, targetnodes, nontargetnodes)
end

function compute_target(::EigMax, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; K, M, Mtarget, Mnontarget, targetnodes, nontargetnodes, Z_full) = ti
    idx = targetconnectedgraphidx(ti)

    Mtarget[:, idx] .= view(M, targetnodes)

    if size(Mnontarget, 1) > 0
        Mnontarget[:, idx] .= view(M, nontargetnodes)
    end
end

update_connectedgraph_output!(output, ::Level, ::EigMax, ::TargetInit, v) = nothing

# We do most of eigmax in finalize_connectedgraph_output!
function finalize_connectedgraph_output!(
    (vˡ, λ, vʳ), ::ConnectedGraphLevel, em::EigMax, cgi::ConnectedGraphInit, intermediates
)
    (; Mtarget, Mnontarget, targetnodes, nontargetnodes) = intermediates


    # size of the full problem
    n = nsources(connectedgraph(cgi))

    # use an Arnoldi based eigensolver to compute the largest
    # (absolute) eigenvalue and right vector (of submatrix)
    Fps = ArnoldiMethod.partialschur(Mtarget; nev=1, tol=em.tol)
    λ₀, vʳ₀ = ArnoldiMethod.partialeigen(Fps[1])

    # Computing the left and right vectors is optional
    if em.right isa Right
        # assign to the full right vector
        vʳ[targetnodes] .= vʳ₀
        vʳ[nontargetnodes] .= Mnontarget * vʳ₀ ./ λ₀[1]
    end

    if em.left isa Left
        # compute left vector (of submatrix) by shift-invert
        F = lu(Mtarget - λ₀[1] * I)
        rng = isnothing(em.seed) ? MersenneTwister() : MersenneTwister(seed)
        # TODO: explain rand here in a comment
        vˡ₀ = ldiv!(F', rand(rng, length(targetnodes))) # This is a hack that ensures a square matrix
        rmul!(vˡ₀, inv(vˡ₀[1]))
        # assign to the full left vector
        vˡ[targetnodes] .= vˡ₀
    end

    # Assign to the output Ref for λ
    λ[] = λ₀[1]

    return vˡ, λ[], vʳ
end

# EdgeBetweenness

# At the GridGraph level we return a Vector or SparseMatrixCSC,
# one for each connected graph (often just one total)
function allocate_output(
    l::GridGraphLevel,
    ::ReturnCustom,
    ::EdgeBetweenness,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    Vector{SparseMatrixCSC{Float64,Int}}(undef, length(connectedgraphs)) => l 
end
# At the ConnectedGraph level we return a SparseMatrixCSC
function allocate_output(
    l::ConnectedGraphLevel,
    ::ReturnCustom,
    ::EdgeBetweenness,
    ::ConScapeProblem,
    ::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    mapnz(_ -> 0.0, precalculation.W) => l
end

function allocate_intermediate(m::EdgeBetweenness, cgi::ConnectedGraphInit)
    (; W) = cgi

    XᵀZ_full = zeros(connectedgraph_size(cgi))
    XdiagZⁱ = zeros(length(targetids(cgi)))

    return (; XdiagZⁱ, XᵀZ_full)
end

# RandomShortestPath / RandomWalk
function compute_target!(
    output,
    ::Union{ConnectedGraphLevel,GridGraphLevel},
    m::EdgeBetweenness,
    ti::TargetInit{<:Union{RSP,RandomWalk}}
)
    (; XdiagZⁱ, XᵀZ_full) = intermediates(ti)
    (; IW_adj_factorization, Z, Zⁱ) = ti
    node = targetnode(ti)
    idx = targetconnectedgraphidx(ti)

    weights = _weight(m, ti)
    XdiagZⁱ[idx] = sum(weights) * Zⁱ[node]
    XZⁱ = workspace(ti) .= weights .* Zⁱ
    XᵀZ = ldiv!(ti, IW_adj_factorization, XZⁱ)
    view(XᵀZ_full, :, idx) .+= XᵀZ

    # We only update output in finalize_connectedgraph_output!
    return output
end

function finalize_connectedgraph_output!(
    output::SparseMatrixCSC,
    ::ConnectedGraphLevel,
    ::EdgeBetweenness,
    cgi::ConnectedGraphInit,
    intermediates
)
    (; W, Z_full, Zrows_full) = cgi # This Z is the full graph size
    (; XdiagZⁱ, XᵀZ_full) = intermediates

    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        node = target.node

        XᵀZ_full[node, :] .-= XdiagZⁱ .* view(Zrows_full, :, node)
    end

    foreachnz(W) do i, j, n
        @inbounds output.nzval[n] =
            # TODO: is j in the right place?
            W.nzval[n] * only(view(Z_full, j, :)' * view(XᵀZ_full, i, :))
    end
    return output
end

# TODO: not copied to a vector?
function transfer_to_gridgraph_output!(
    dest::SparseMatrixCSC,
    ::GridGraphLevel,
    source::SparseMatrixCSC,
    ::ConnectedGraphLevel,
    m::EdgeBetweenness,
    cgi::ConnectedGraphInit
)
    I = LinearIndices(size(cgi))[sourceids(cgi)]
    J = map(t -> t.gridgraphidx, targetids(cgi))
    dest[I, J] .= source

    return dest
end



# LeastCostPath EdgeBetweenness
# Not implemented


######################################################################################
# Proximities

function compute_target(::Distance, ti::TargetInit{<:Euclidean})
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))
    return ti.workspace .= _hypot.(sourceids(ti), (targetspatialidx(ti),))
end
compute_target(::Distance, ti::TargetInit{<:LCP}) = readonlyarray(ti.shortest_paths.dists)

function compute_target(
    ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
)
    (; IW_factorization) = ti
    node = targetnode(ti)
    PC_rowsums = ti.workspace
    # Set target rowsum of PC to zero
    PC_rowsums .= ti.PC_rowsums
    PC_rowsums[node] = 0
    # Solve (I - W) \ PC_rowsums
    return ldiv!(ti, IW_factorization, PC_rowsums)
end

# RSP
function compute_target(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Y, Zⁱ) = ti
    C̄ = workspace(ti) .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[targetnode(ti)]
    return C̄
end
function compute_target(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    return workspace .= -log.(max.(0, sp)) ./ θ
end
function compute_target(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    return workspace .= sp .^ (1 / θ)
end
function compute_target(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z, workspace) = ti
    return workspace .= Z ./ Z[targetnode(ti)]
end


############################################################################################
# Mean Kullback-Leibler Divergence

function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:LCP})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    node = targetnode(ti)
    output = ti.workspace
    from = Vector{Int}(undef, length(output))
    to = Vector{Int}(undef, length(output))

    n = length(from)
    dsp = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, node)
    parents = dsp.parents
    # TODO explain why this is needed
    parents[node] = node

    # Initialise arrays
    fill!(output, 0.0)
    from .= 1:n
    to .= parents

    # TODO explain what this loop does
    while true
        notdone = false
        for i in 1:n
            fromᵢ, toᵢ = from[i], to[i]
            notdone |= (fromᵢ != toᵢ)
            fromᵢ == toᵢ && continue
            output[i] += -log(P[fromᵢ, toᵢ])
            from[i] = parents[toᵢ]
        end
        if !notdone
            break
        end
        from, to = to, from
    end
    return sum(output .*= qˢ) * qᵗ # qs' * output * qt
end
function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:RandomWalk})
    return 0.0 # Trivially returns zero
end
function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    (; θ, qˢ, qᵗ, workspace) = ti
    fed = get_or_compute_target!(ti, FreeEnergyDistance())
    ec = get_or_compute_target!(ti, ExpectedCost())
    diff = workspace .= fed .- ec
    return sum(diff .*= qˢ) * qᵗ * θ # qˢ' * diff * qᵗ * θ
end

# What are these, how are they different to the RSP versions?
# compute_target(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute_target(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

###########################################################################################
# FunctionalHabitat

compute_target(::FunctionalHabitat, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

# This differs form FunctionalHabitat in that it returns the full size matrix
compute_target(::LandscapeMatrix, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

###########################################################################################
# Betweenness

# LeastCostPath
function compute_target(m::Betweenness, ti::TargetInit{<:LCP})
    (; shortest_paths, path_allocs, workspace) = ti
    node = targetnode(ti)
    shortest_paths_enumerated = 
        Graphs.enumerate_paths!(path_allocs, shortest_paths, 1:length(path_allocs))
    # Set the target path to only contain itself
    targetpath = resize!(shortest_paths_enumerated[node], 1)
    targetpath[1] = node
    # Get the target weights
    weights = _weight(m, ti)
    btw = workspace .= 0.1

    @inbounds for s in eachindex(sourceids(ti))
        w = weights[s]
        for p in shortest_paths_enumerated[s]
            btw[p] += w
        end
    end
    return btw
end
# RandomShortestPath / RandomWalk (differences are only in IW and weights)
function compute_target(m::Betweenness, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; Z, Zⁱ, IW_adj_factorization, workspace) = ti
    weight = _weight(m, ti)
    node = targetnode(ti)
    isnothing(weight) && error("Betweenness weight is `nothing`")
    XZⁱt = workspace .= weight .* Zⁱ
    # Find the scaling factor: if any of XZⁱ is above 1.0 there is a risk of Inf overflow
    λ = max(1.0, maximum(XZⁱt))
    # TODO: explain what this subtraction does
    XZⁱt[node] -= Zⁱ[node] * sum(weight)
    # Scale MZⁱ with λ
    XZⁱtλ = XZⁱt .*= inv(λ)
    # Solve (I - W)' \ MZⁱλ, then multiply by Z and λ scaling
    return ldiv!(ti, IW_adj_factorization, XZⁱtλ) .*= λ .* Z
end

_weight(m::Union{EdgeBetweenness,Betweenness}, ti::TargetInit) =
    _weight(weighting(m), ti)
_weight(::Unweighted, ti::TargetInit) = 1
_weight(::ProximityWeighted, ti::TargetInit) = ti.K
_weight(::QualityAndProximityWeighted, ti::TargetInit) = ti.M
_weight(::QualityWeighted, ti::TargetInit) = ti.Q
_weight(w::CustomWeighted, ti::TargetInit) = w.weight

# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::FunctionalHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * sourcequality_spatial(init)[target] * targetquality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0

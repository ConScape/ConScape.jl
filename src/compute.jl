@inline function get_or_compute!(ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)
    haskey(st, x) && return st[x]
    val = compute(m, ti)
    if returntrait(m) isa ReturnDenseSpatial
        st[x] = readonlyarray(val) 
        return readonlyarray(val)
    else
        return val
    end
end
@inline function get_or_compute!(ti::TargetInit, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]

    val = target_precalculation!(ti, x)
    st[x] = readonlyarray(val)
    return val
end


@inline function compute_and_update_output!(output::Pair, ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)

    if haskey(st, x)
        update_output!(output, m, t, st[x])
    else
        val = compute!(output, m, ti)
        # Store simple array outputs
        if val isa Array
            st[x] = readonlyarray(val)
        end
    end

    return output
end

# Most measures dont need to deal with `update_output` and just use `compute`
function compute!(output, m::Measure, ti)
    val = compute(m, ti)
    update_output!(output, m, ti, val)

    return output
end

#------------------------------------------------------------------------------------------
# ConnectedGraph level measures

# Compute Target level measures for a connected subgraph
# function compute(m::Measure, cgi::ConnectedGraphInit)
#     output = allocate_output(ConnectedGraphLevel(), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))
#     for target in targetids(cgi)
#         ti = TargetInit(cgi, target)
#         compute!(output, m, ti)
#     end
#     return finalize_output!(output, m, cgi)
# end

# Allocate sqauare matrix of target * target size
allocate_output(l::GridGraphLevel, ::ReturnCustom, m::EigMax, ::ConScapeProblem, ::GridGraph, connectedgraphs::Vector) = 
    l => Vector{Tuple{Vector{Float64},Array{Float64,0},Vector{Float64}}}(undef, length(connectedgraphs))
function allocate_output(l::Level, ::ReturnCustom, m::EigMax, ::ConScapeProblem, ::GridGraph, cg::ConnectedGraph, precalculation)
    n = length(sourceids(cg))
    vʳ = fill(NaN, n)
    λ = fill(0.0)
    vˡ = zeros(n)
    return l => (vʳ, λ, vˡ)
end

function allocate_intermediate(::EigMax, cgi::ConnectedGraphInit)
    m = length(targetnodes(cgi))
    targetnodes = map(x -> x.node, targetids(cgi))
    nontargetnodes = setdiff(1:n, targetnodes)
    Mtarget = zeros(m, m)
    Mnontarget = zeros(length(nontargetnodes), m)
    return (; Mtarget, Mnontarget, targetnodes, nontargetnodes)
end

function compute!(output::Pair, ::EigMax, ti::TargetInit)
    (; Mtarget, Mnontarget, M) = ti

    # We need targets and non-targets in separate matrices.
    # This is still a lot of allocations, but the least possible for EigMax
    M₀₀[target(ti).node, :] .= view(M, targetnodes)
    Mnontarget[target(ti).node, :] .= view(M, nontargetnodes)

    return output
end

# We do most of eigmax in finalize_output
function finalize_output!((vˡ, λ, vʳ), ::Level, ::EigMax, cgi::ConnectedGraphInit)
    (; Mtarget, Mnontarget, targetnodes, nontargetnodes) = ti

    tol = 1e-14 # TODO as a keyword somewhere

    # size of the full problem
    n = nsources(connectedgraph(cgi))

    # use an Arnoldi based eigensolver to compute the largest 
    # (absolute) eigenvalue and right vector (of submatrix)
    Fps = ArnoldiMethod.partialschur(Mtarget; nev=1, tol)
    λ₀, vʳ₀ = ArnoldiMethod.partialeigen(Fps[1])

    # compute left vector (of submatrix) by shift-invert
    Flu = lu(Mtarget - λ₀[1] * I)
    # TODO: explain rand here in a comment
    vˡ₀ = ldiv!(Flu', rand(length(targetnodes)))
    rmul!(vˡ₀, inv(vˡ₀[1]))

    # assign to the full right vector
    vʳ[targetnodes] = vʳ₀
    vʳ[nontargetnodes] = Mnontarget * vʳ₀ / λ₀[1]

    # assign to the full left vector
    vˡ[targetnodes] .= vˡ₀
    λ[] = λ₀[1]

    return vˡ, λ[], vʳ
end

# EdgeBetweenness

allocate_output(l::GridGraphLevel, ::ReturnCustom, ::EdgeBetweenness, ::ConScapeProblem, ::GridGraph, connectedgraphs::Vector) = 
    l => Vector{SparseMatrixCSC{Float64,Int}}(undef, length(connectedgraphs))
allocate_output(l::ConnectedGraphLevel, ::ReturnCustom, ::EdgeBetweenness, ::ConScapeProblem, ::GridGraph, ::ConnectedGraph, precalculation) = 
    l => mapnz(_ -> 0.0, precalculation.W)

function allocate_intermediate(m::EdgeBetweenness, cgi::ConnectedGraphInit)
    (; W) = cgi

    store[] = (; 
        W,
        M=zeros(size(W)),
        K=zeros(size(W)),
        Z=zeros(size(W)),
        Zⁱ=zeros(size(W)),
        Zrows=zeros(size(W)),
        XZⁱ=zeros(size(W)),
        XᵀZ=zeros(size(W)),
        XᵀZminusdiag=zeros(size(W)),
    )

    Zmatrix = zeros(size(W))
    XᵀZmatrix = zeros(size(W))
    XdiagZⁱ = zeros(length(targetids(cgi)))

    return (; Zmatrix, XdiagZⁱ, XᵀZmatrix)
end

# RandomShortestPath / RandomWalk
function compute!(
    output::Pair, m::EdgeBetweenness, ti::TargetInit{<:Union{RSP,RandomWalk}}
)
    (; IW_adj_factorization, Z, Zⁱ) = ti
    (; Zmatrix, XdiagZⁱ, XᵀZmatrix) = intermediates(ti)
    node = target(ti).node

    Zmatrix[:, node] .= Z
    weights = _weight(m, ti)
    XdiagZⁱ[node] = sum(weights) * Zⁱ[node]
    XZⁱ = workspace(ti) .= weights .* Zⁱ
    XᵀZ = ldiv!(ti, IW_adj_factorization, XZⁱ)
    store[].XᵀZ[:, node] .= XᵀZ
    XᵀZmatrix[:, node] .+= XᵀZ

    store[].M[:, node] .= weights
    store[].Z[:, node] .= Z
    store[].Zⁱ[:, node] .= Zⁱ
    store[].XZⁱ[:, node] .= XZⁱ

    # We only update output in finalize_output!
    return output
end

function finalize_output!(
    output::SparseMatrixCSC, ::Level, ::EdgeBetweenness, cgi::ConnectedGraphInit, intermediates
)
    (; W) = cgi
    (; Zmatrix, XdiagZⁱ, XᵀZmatrix) = intermediates

    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        (; Zrows) = ti
        node = target.node
        store[].Zrows[:, node] .= Zrows

        XᵀZmatrix[node, :] .-= XdiagZⁱ .* Zrows
    end

    foreachnz(W) do i, j, n
        @inbounds output.nzval[n] = 
            W.nzval[n] * only(view(Zmatrix, j, :)' * view(XᵀZmatrix, i, :))
    end
    store[] = merge(store[], intermediates, (; Z=Zmatrix, bet_edge_k=output, XᵀZmatrix))
    return output
end

function transfer_output!(
    dest::SparseMatrixCSC, source::SparseMatrixCSC, m::EdgeBetweenness, cgi::ConnectedGraphInit
    )
    dest[LinearIndices(size(cgi))[sourceids(cgi)], map(t -> t.gridgraphidx, targetids(cgi))] .= source
end



# LeastCostPath EdgeBetweenness
# Not implemented


######################################################################################
# Proximities

function compute(::Distance, ti::TargetInit{<:Euclidean})
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))
    return ti.workspace .= _hypot.(sourceids(ti), (target(ti).spatialidx,))
end
compute(::Distance, ti::TargetInit{<:LCP}) =
    readonlyarray(ti.shortest_paths.dists)

function compute(
    ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
)
    (; IW_factorization) = ti
    node = target(ti).node
    PC_rowsums = ti.workspace
    # Set target rowsum of PC to zero
    PC_rowsums .= ti.PC_rowsums
    PC_rowsums[node] = 0
    # Solve (I - W) \ PC_rowsums
    return ldiv!(ti, IW_factorization, PC_rowsums)
end

# RSP
function compute(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Y, Zⁱ) = ti
    C̄ = ti.workspace .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[target(ti).node]
    return C̄
end
function compute(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute!(ti, SurvivalProbability())
    return workspace .= -log.(max.(0, sp)) ./ θ
end
function compute(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute!(ti, SurvivalProbability())
    return workspace .= sp .^ (1 / θ)
end
function compute(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z, workspace) = ti
    return workspace .= Z ./ Z[target(ti).node]
end


######################################################################################
# Mean Kullback-Leibler Divergence

function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:LCP})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    output = ti.workspace
    from = Vector{Int}(undef, length(output))
    to = Vector{Int}(undef, length(output))

    n = length(from)
    dsp = Graphs.dijkstra_shortest_paths(cost_weighted_digraph, target(ti).node)
    parents = dsp.parents
    # TODO explain why this is needed
    parents[target(ti).node] = target(ti).node

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
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:RandomWalk})
    return 0.0 # Trivially returns zero
end
function compute(::KullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    (; θ, qˢ, qᵗ, workspace) = ti
    fed = get_or_compute!(ti, FreeEnergyDistance())
    ec = get_or_compute!(ti, ExpectedCost())
    diff = workspace .= fed .- ec
    return sum(diff .*= qˢ) * qᵗ * θ # qˢ' * diff * qᵗ * θ
end

# What are these, how are they different to the RSP versions?
# compute(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

######################################################################################
# FunctionalHabitat 

compute(::FunctionalHabitat, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

# This differs form FunctionalHabitat in that it returns the full size matrix
compute(::LandscapeMatrix, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

######################################################################################
# Betweenness

# LeastCostPath
function compute(m::Betweenness, ti::TargetInit{<:LCP})
    (; shortest_paths, path_allocs, workspace) = ti
    node = target(ti).node
    shortest_paths_enumerated = Graphs.enumerate_paths!(path_allocs, shortest_paths, 1:length(path_allocs))
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
function compute(m::Betweenness, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; Z, Zⁱ, IW_adj_factorization, workspace) = ti
    weight = _weight(m, ti)
    isnothing(weight) && error("Betweenness weight is `nothing`")
    XZⁱt = workspace .= weight .* Zⁱ
    # Find the scaling factor: if any of XZⁱ is above 1.0 there is a risk of Inf overflow
    λ = max(1.0, maximum(XZⁱt))
    # TODO: explain what this subtraction does
    XZⁱt[target(ti).node] -= Zⁱ[target(ti).node] * sum(weight)
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

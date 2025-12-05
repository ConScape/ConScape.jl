"""
    EdgeBetweenness <: GraphMeasure

    EdgeBetweenness(weighting)

Compute betweenness of all edges weighted by qualities of 
source s and target t and the proximity between s and t. 

$WEIGHTING_ARGUMENT

Returns a sparse matrix where element (i, j) is the betweenness of edge (i, j).
"""
@kwdef struct EdgeBetweenness{W} <: GraphMeasure
    weighting::W
end

weighting(gm::EdgeBetweenness) = gm.weighting
needs_workspaces(::EdgeBetweenness) = 4
returntrait(::EdgeBetweenness) = ReturnCustom()

Base.Symbol(m::EdgeBetweenness) = Symbol(nameof(typeof(m)), :_, nameof(typeof(weighting(m))))

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


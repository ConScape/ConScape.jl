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
returntrait(::EdgeBetweenness) = ReturnCustom()
needs_workspaces(::EdgeBetweenness) = 4
needs_full_fundamentalmatrix(::EdgeBetweenness, ::RSP) = true
needs_full_fundamentalrowmatrix(::EdgeBetweenness, ::RSP) = true

Base.Symbol(m::EdgeBetweenness) = Symbol(nameof(typeof(m)), :_, nameof(typeof(weighting(m))))

# At the GridGraph level we return a Vector of SparseMatrixCSC,
# one for each connected graph (often just one total)
function allocate_gridgraph_output(
    ::ReturnCustom,
    m::EdgeBetweenness,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    # Make a Vector of SparseMatrixCSC
    o = Vector{SparseMatrixCSC{Float64,Int}}(undef, length(connectedgraphs))
    return MeasureOutput(m, o)
end
# At the ConnectedGraph level we return a SparseMatrixCSC
function allocate_connectedgraph_output(
    l::ConnectedGraphLevel,
    ::ReturnCustom,
    m::EdgeBetweenness,
    ::ConScapeProblem,
    ::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    # Make a zeroed sparse matrix with the same pattern as W
    o = mapnz(_ -> 0.0, precalculation.W)
    return MeasureOutput(m, o)
end

function compute_connectedgraph!(output, m::EdgeBetweenness, cgi::ConnectedGraphInit)
    (; W, Z_full, Zrows_full) = cgi # This Z is the full graph size

    XᵀZ_full = mworkspace(cgi)
    XdiagZⁱ::VDe = fill!(view(workspace(cgi), 1:ntargets(cgi)), 0.0)

    # Loop over targets to calculate the diagonal
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        idx = target.connectedgraphidx
        (; IW_adj_factorization, Z, Zⁱ) = ti
        node = targetnode(ti)
        idx = targetconnectedgraphidx(ti)

        weights = _weight(m, ti)
        XdiagZⁱ[idx] = sum(weights) * Zⁱ[node]
        XZⁱ = workspace(ti) .= weights .* Zⁱ
        XᵀZ = ldiv!(ti, IW_adj_factorization, XZⁱ)
        @views XᵀZ_full[:, idx] .+= XᵀZ
    end

    # Loop over targets to update XᵀZ  
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        node = target.node

        view(XᵀZ_full, node, :) .-= XdiagZⁱ .* view(Zrows_full, :, node)
    end

    foreachnz(W) do i, j, n
        @inbounds output.nzval[n] =
            # TODO: is j in the right place?
            W.nzval[n] * only(view(Z_full, j, :)' * view(XᵀZ_full, i, :))
    end
    put!(mworkspaces(cgi), XᵀZ_full)

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
    # TODO: this should be a Vector{Int} already?
    J = map(t -> t.gridgraphidx, targetids(cgi))
    dest[I, J] .= source

    return dest
end

# LeastCostPath EdgeBetweenness
# Not implemented


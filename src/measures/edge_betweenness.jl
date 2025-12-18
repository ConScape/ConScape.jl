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

computelevel(::EdgeBetweenness) = ConnectedGraphLevel()
returntrait(::EdgeBetweenness) = ReturnAssignedSparse()

num_vector_workspaces(::EdgeBetweenness, ::LandscapeMovement) = 4
needs_full_fundamentalmatrix(::EdgeBetweenness, ::RSP) = true
needs_full_fundamentalrowmatrix(::EdgeBetweenness, ::RSP) = true
needs_edgebetweenness_workspace(::EdgeBetweenness, ::RSP) = true

@generated Base.Symbol(m::EdgeBetweenness{W}) where W = 
    QuoteNode(Symbol(nameof(m), :_, nameof(W)))

# At the ConnectedGraph level we return a SparseMatrixCSC
function allocate_connectedgraph_output(
    l::ConnectedGraphLevel,
    ::ReturnSparseGraph,
    m::EdgeBetweenness,
    ::GridGraph,
    cg::ConnectedGraph,
)
    # Make a zeroed sparse matrix with the same pattern as W
    # This is faster than starting with an empty sparse matrix
    o = mapnz(_ -> 0.0, cg.W)
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
        (; F_IW_adj, Z, Zⁱ) = ti
        node = targetnode(ti)
        idx = targetconnectedgraphidx(ti)

        weights = _weight(m, ti)
        XdiagZⁱ[idx] = sum(weights) * Zⁱ[node]
        XZⁱ = workspace(ti) .= weights .* Zⁱ
        XᵀZ = ldiv!(ti, F_IW_adj, XZⁱ)
        @views XᵀZ_full[:, idx] .= XᵀZ
    end

    # Loop over targets to update XᵀZ  
    for node in 1:nsources(cgi)
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

# LeastCostPath EdgeBetweenness
# Not implemented


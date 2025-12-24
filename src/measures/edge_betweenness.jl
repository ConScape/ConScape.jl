const store = Ref{Any}((;))

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
returntrait(::EdgeBetweenness) = ReturnCustomSparse()

num_vec_workspaces(::EdgeBetweenness, ::LandscapeMovement) = 4
needs_full_fundamentalmatrix(::EdgeBetweenness, ::RSP) = true
needs_full_fundamentalrowmatrix(::EdgeBetweenness, ::RSP) = true
needs_edgebetweenness_workspace(::EdgeBetweenness, ::RSP) = true

@generated Base.Symbol(m::EdgeBetweenness{W}) where W =
    QuoteNode(Symbol(nameof(m), :_, nameof(W)))

# At the ConnectedGraph level we return a SparseMatrixCSC
function allocate_connectedgraph_output(
    l::Union{ConnectedGraphLevel,TargetLevel},
    ::ReturnCustomSparse,
    m::EdgeBetweenness,
    ::GridGraph,
    cg::ConnectedGraph,
)
    # Make a zeroed sparse matrix with the same pattern as W
    # This is faster than starting with an empty sparse matrix
    o = mapnz(_ -> 0.0, steplikelihood(cg))
    return MeasureOutput(m, o)
end

# RSP/RandomWalk
function compute_connectedgraph!(output, m::EdgeBetweenness, cgi::ConnectedGraphInit{<:Union{<:RSP,<:RandomWalk}})
    (; W, Z_full, Zrows_full) = cgi # This Z is the full graph size

    XᵀZ_full = mat_workspace(cgi)
    XZⁱ_full = copy(XᵀZ_full)
    M_full = copy(XᵀZ_full)
    XZⁱ = vec_workspace(cgi)
    XᵀZ = vec_workspace(cgi)
    XdiagZⁱ = copy(fill!(view(vec_workspace(cgi), 1:ntargets(cgi)), 0.0))

    # Loop over targets to calculate the diagonal
    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        idx = target.connectedgraphidx
        (; F_IW_adj, Z, Zⁱ) = ti
        node = targetnode(ti)
        idx = targetconnectedgraphidx(ti)

        weights = _weight(m, ti)
        XdiagZⁱ[idx] = sum(weights) * Zⁱ[node]
        XZⁱ .= weights .* Zⁱ
        M_full[:, idx] .= weights
        XZⁱ_full[:, idx] .= XZⁱ
        copy!(XᵀZ, XZⁱ)
        ldiv!(solver(ti), XᵀZ, F_IW_adj, XZⁱ)
        @views XᵀZ_full[:, idx] .= XᵀZ
    end

    # Loop over targets to update XᵀZ
    for node in 1:nsources(cgi)
        view(XᵀZ_full, node, :) .-= XdiagZⁱ .* view(Zrows_full, :, node)
    end

    foreachnz(W) do i, j, n
        output.nzval[n] =
            W.nzval[n] * only(view(Z_full, j, :)' * view(XᵀZ_full, i, :))
    end

    store[] = merge(store[], (; W, M=M_full, Z=Z_full, Zrows=Zrows_full, XdiagZⁱ=copy(XdiagZⁱ), XᵀZ=copy(XᵀZ_full), XZⁱ=XZⁱ_full))

    put!(mat_workspaces(cgi), XᵀZ_full)

    return output
end

# LeastCostPath
# Not implemented

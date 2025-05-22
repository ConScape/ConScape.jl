# Simulate a permeable wall
function permeable_wall_sim(nrows::Int, ncols::Int;
    scaling::Float64=0.5,
    wallwidth::Integer=3,
    wallposition::Float64=0.5,
    corridorwidths::NTuple{<:Any,Int}=(3, 3),
    corridorpositions=(0.35, 0.7),
    impossible_likelihood::Real=1e-20,
    nhood_size::Integer=8,
    kw...
)
    # 1. initialize landscape
    transitionlikelihood = _generate_likelihood(nrows, ncols, nhood_size) .* scaling
    gridgraph = GridGraph(; transitionlikelihood, kw...)

    # # 2. compute the wall
    wpt = round(Int, ncols * wallposition - wallwidth/2 + 1)
    xs  = range(wpt, stop=wpt + wallwidth - 1)

    # 3. compute the corridors
    ys = Int[]
    for i in 1:length(corridorwidths)
        cpt = floor(Int, nrows * corridorpositions[i]) - ceil(Int, corridorwidths[i]/2)
        if i == 1
            append!(ys, 1:cpt)
        else
            append!(ys, range(maximum(ys) + 1 + corridorwidths[i-1], stop=cpt))
        end
    end
    append!(ys, range(maximum(ys) + 1 + corridorwidths[end]  , stop=nrows))

    impossible_nodes = vec(CartesianIndex.(collect(Iterators.product(ys, xs))))
    return _set_impossible_nodes(gridgraph, impossible_nodes, impossible_likelihood)
end

#=
Generate the affinity matrix of a grid graph, where each
pixel is connected to its vertical and horizontal neighbors.

Parameters:
- nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
=#
function _generate_likelihood(nrows, ncols, nhood_size)
    nhood_size in (4, 8) || throw(ArgumentError("nhood_size must be either 4 or 8"))

    likelihood = kron(spdiagm(0 => ones(ncols)), spdiagm(-1 => ones(nrows - 1), 1 => ones(nrows - 1))) +
                 kron(spdiagm(-1 => ones(ncols - 1), 1 => ones(ncols - 1)), spdiagm(0=>ones(nrows)))

    if nhood_size == 8
        likelihood .+= kron(spdiagm(-1 => ones(ncols - 1), 1 => ones(ncols - 1)),
                            spdiagm(-1 => fill(1/√2, nrows - 1), 1 => fill(1/√2, nrows - 1)))
    end

    return likelihood
end

#=
Make pixels impossible to move to by changing the affinities to them to zero.
Input:
    - node_list: list of nodes (either node_ids or coordinate-tuples) to be made impossible
=#
function _set_impossible_nodes(g::GridGraph, node_list::Vector{CartesianIndex{2}}, impossible_affinity=1e-20)
    # Find the indices of the coordinates in the source_ids vector
    node_list_idx = [findfirst(isequal(n), sourceids(g))::Int for n in node_list]

    # Copy affinities and qualities for modification
    transitionlikelihood = copy(g.transitionlikelihood)
    sourcequality = copy(g.sourcequality)
    targetquality = copy(g.targetquality)

    # Set (nonzero) values to impossible_affinity:
    # affinitymatrix
    # FIXME! Row slicing of a sparse matrix is really inefficient
    transitionlikelihood[node_list_idx, :] = impossible_affinity .* (transitionlikelihood[node_list_idx, :] .> 0)
    transitionlikelihood[:, node_list_idx] = impossible_affinity .* (transitionlikelihood[:, node_list_idx] .> 0)

    dropzeros!(transitionlikelihood)

    # Qualities
    sourcequality[node_list] .= 0
    targetquality[node_list] .= 0

    # Generate a new Grid based on the modified affinitymatrix
    return GridGraph(; transitionlikelihood, transitioncost=g.transitioncost, sourcequality, targetquality)
end
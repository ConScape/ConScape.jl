# simulate a permeable wall
function perm_wall_sim(size::Tuple{Int,Int};
    scaling::Float64=0.5,
    wallwidth::Integer=3,
    wallposition::Float64=0.5,
    corridorwidths::NTuple{<:Any,Int}=(3,3),
    corridorpositions=(0.35,0.7),
    impossible_affinity::Real=1e-20,
    nhood_size::Integer=8,
    kwargs...
)

    # 1. initialize landscape
    affinities = _generate_affinities(nrows, ncols, nhood_size)
    g = Grid(nrows, ncols; affinities=affinities*scaling, kwargs...)

    # # 2. compute the wall
    wpt = round(Int, ncols*wallposition - wallwidth/2 + 1)
    xs  = range(wpt, stop=wpt + wallwidth - 1)

    # 3. compute the corridors
    ys = Int[]
    for i in 1:length(corridorwidths)
        cpt = floor(Int, nrows*corridorpositions[i]) - ceil(Int, corridorwidths[i]/2)
        if i == 1
            append!(ys, 1:cpt)
        else
            append!(ys, range(maximum(ys) + 1 + corridorwidths[i-1], stop=cpt))
        end
    end
    append!(ys, range(maximum(ys) + 1 + corridorwidths[end]  , stop=nrows))

    impossible_nodes = vec(CartesianIndex.(collect(Iterators.product(ys, xs))))
    g = _set_impossible_nodes(g, impossible_nodes, impossible_affinity)

    return g
end

⊗ = kron
#=
Generate the affinity matrix of a grid graph, where each
pixel is connected to its vertical and horizontal neighbors.

Parameters:
- nhood_size: 4 creates horizontal and vertical edges, 8 creates also diagonal edges
=#
function _generate_affinities(nrows, ncols, nhood_size)

    if !(nhood_size ∈ (4, 8))
        throw(ArgumentError("nhood_size must be either 4 or 8"))
    end

    affinities = spdiagm(0=>ones(ncols)) ⊗ spdiagm(-1=>ones(nrows - 1), 1=>ones(nrows - 1)) +
        spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗ spdiagm(0=>ones(nrows))

    if nhood_size == 8
        affinities .+= spdiagm(-1=>ones(ncols - 1), 1=>ones(ncols - 1)) ⊗
              spdiagm(-1=>fill(1/√2, nrows - 1), 1=>fill(1/√2, nrows - 1))
    end

    return affinities
end

const N4 = (( 0, -1, 1.0),
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            ( 0,  1, 1.0))
const N8 = ((-1, -1,  √2),
            ( 0, -1, 1.0),
            ( 1, -1,  √2),
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            (-1,  1,  √2),
            ( 0,  1, 1.0),
            ( 1,  1,  √2))

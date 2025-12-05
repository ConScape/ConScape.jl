# Betweenness

abstract type BetweennessWeighting end

struct Unweighted <: BetweennessWeighting end

"""
    QualityWeighted <: BetweennessWeighting

    QualityWeighted()

Compute betweenness of nodes or edges weighted by source and target qualities.
"""
struct QualityWeighted <: BetweennessWeighting end

"""
    ProximityWeighted <: BetweennessWeighting

    ProximityWeighted()

Compute betweenness of nodes or edges weighted by the 
proxmimity between source qualities s and target qualities t.
"""
struct ProximityWeighted <: BetweennessWeighting end

"""
    QualityAndProximityWeighted <: BetweennessWeighting

    QualityAndProximityWeighted()

Compute betweenness of nodes or edges weighted by source qualities s 
and target qualities t, and the proximity between s and t.
"""
struct QualityAndProximityWeighted <: BetweennessWeighting end

"""
    CustomWeighted <: BetweennessWeighting

    CustomWeighted(weight)

Holds and arbitrary array of custom weights. 

Used internally.
"""
struct CustomWeighted{W} <: BetweennessWeighting 
    weight::W
end

const WEIGHTING_ARGUMENT = """
- `weighting`: a [`BetweennessWeighting`](@ref): `Unweighted()`, `QualityWeighted()` 
    `ProximityWeighted()` or `QualityAndProximityWeighted()`
"""

"""
    Betweenness <: SpatialMeasure
    
    Betweenness(weighting)

Compute betweenness of all edges weighted by qualities of 
source s and target t and the proximity between s and t,
as defined by the [`MovementMode`](@ref)).

## Arguments

$WEIGHTING_ARGUMENT

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
struct Betweenness{W} <: SpatialMeasure
    weighting::W
end
Betweenness{W}() where W = Betweenness(W())
Betweenness(; weighting) = Betweenness(weighting)

const MovementFlow = Betweenness{QualityAndProximityWeighted}

weighting(gm::Betweenness) = gm.weighting
needs_workspaces(::Betweenness) = 2

Base.Symbol(m::Betweenness) = Symbol(nameof(typeof(m)), :_, nameof(typeof(weighting(m))))

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


# TODO: docstring
struct KullbackLeiblerDivergence <: PathDistributionMeasure end

computelevel(::KullbackLeiblerDivergence) = TargetLevel()
returntrait(::KullbackLeiblerDivergence) = ReturnScalarSum()

# LeastCostPath
function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:LCP})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    node = targetnode(ti)
    output = workspace(ti)
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
# RandomWalk
function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:RandomWalk})
    return 0.0 # Trivially returns zero
end
# RSP
function compute_target(::KullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    (; θ, qˢ, qᵗ) = ti
    fed = get_or_compute_target!(ti, FreeEnergyDistance())
    ec = get_or_compute_target!(ti, ExpectedCost())
    diff = workspace(ti) .= fed .- ec
    return sum(diff .*= qˢ) * qᵗ * θ # qˢ' * diff * qᵗ * θ
end

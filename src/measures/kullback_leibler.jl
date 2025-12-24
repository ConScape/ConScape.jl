
# TODO: docstring
struct MeanKullbackLeiblerDivergence <: PathDistributionMeasure end

computelevel(::MeanKullbackLeiblerDivergence) = TargetLevel()
returntrait(::MeanKullbackLeiblerDivergence) = ReturnScalarSum()
num_vec_workspaces(::MeanKullbackLeiblerDivergence, ::LCP) = 1
num_vec_workspaces(::MeanKullbackLeiblerDivergence, ::RandomWalk) = 0
# RSP: calls FreeEnergyDistance (2) + ExpectedCost (1) + 1 for itself = 4
num_vec_workspaces(::MeanKullbackLeiblerDivergence, ::RSP) = 4

# LeastCostPath
function compute_target(::MeanKullbackLeiblerDivergence, ti::TargetInit{<:LCP})
    (; cost_weighted_digraph, P, qˢ, qᵗ) = ti
    node = targetnode(ti)
    output = vec_workspace(ti)
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
function compute_target(::MeanKullbackLeiblerDivergence, ti::TargetInit{<:RandomWalk})
    return 0.0 # Trivially returns zero
end
# RSP
function compute_target(::MeanKullbackLeiblerDivergence, ti::TargetInit{<:RSP})
    (; θ, qˢ, qᵗ) = ti
    fed = get_or_compute_target!(ti, FreeEnergyDistance())
    ec = get_or_compute_target!(ti, ExpectedCost())
    diff = vec_workspace(ti) .= fed .- ec

    # qˢ' * diff * qᵗ * θ
    return sum(diff .*= qˢ) * qᵗ * θ
end

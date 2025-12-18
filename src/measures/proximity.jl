
# All ProximityMeasure are grouped here as they are relatively trivial

# All ProximityMeasure compute at target level, 
# and return a SparseArray of single s/t values
computelevel(::ProximityMeasure) = TargetLevel()
returntrait(::ProximityMeasure) = ReturnAssignedSparse()

# TODO: docs
struct Distance <: DistanceMeasure end
struct ExpectedCost <: DistanceMeasure end
struct FreeEnergyDistance <: DistanceMeasure end
struct HittingTime <: DistanceMeasure end

# Not conditional upon arrival
struct PowerMeanProximity <: ProximityMeasure end
struct SurvivalProbability <: ProximityMeasure end

function compute_target(::Distance, ti::TargetInit{<:Euclidean})::RVDe
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))

    targetcol = workspace(ti) .= _hypot.(sourceids(ti), (targetspatialidx(ti),))
    return readonlyarray(targetcol)
end
compute_target(::Distance, ti::TargetInit{<:LCP})::RVDe = readonlyarray(ti.shortest_paths.dists)

# function compute_target(
#     ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
# )::VDe
#     (; IW_factorization, PC_rowsums) = ti
#     node = targetnode(ti)
#     # Set target rowsum of PC to zero
#     PC_rowsums = workspace(ti) .= ti.
#     PC_rowsums[node] = 0
#     # Solve (I - W) \ PC_rowsums
#     return ldiv!(ti, IW_factorization, PC_rowsums)
# end

# RSP
function compute_target(::ExpectedCost, ti::TargetInit{<:RSP})::RVDe
    (; Y, Zⁱ) = ti
    C̄ = workspace(ti) .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[targetnode(ti)]
    return readonlyarray(C̄)
end
function compute_target(::FreeEnergyDistance, ti::TargetInit{<:RSP})::RVDe
    (; θ) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    targetcol = workspace(ti) .= -log.(max.(0, sp)) ./ θ
    return readonlyarray(targetcol)
end
function compute_target(::PowerMeanProximity, ti::TargetInit{<:RSP})::RVDe
    (; θ) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    targetcol = workspace(ti) .= sp .^ (1 / θ)
    return readonlyarray(targetcol)
end
function compute_target(::SurvivalProbability, ti::TargetInit{<:RSP})::RVDe
    (; Z) = ti
    targetcol = workspace(ti) .= Z ./ Z[targetnode(ti)]
    return readonlyarray(targetcol)
end

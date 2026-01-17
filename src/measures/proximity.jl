
# All ProximityMeasure are grouped here as they are relatively trivial

# All ProximityMeasure compute at target level,
# and return a SparseArray of single s/t values
computelevel(::ProximityMeasure) = TargetLevel()
returntrait(::ProximityMeasure) = ReturnAssignedDense()
# Most proximity measures use 1 vec_workspace
num_vec_workspaces(::ProximityMeasure, ::LandscapeMovement) = 1

# TODO: docs
struct Distance <: DistanceMeasure end
struct ExpectedCost <: DistanceMeasure end
struct FreeEnergyDistance <: DistanceMeasure end
struct HittingTime <: DistanceMeasure end

# Not conditional upon arrival
struct PowerMeanProximity <: ProximityMeasure end
struct SurvivalProbability <: ProximityMeasure end

# These call SurvivalProbability internally, so need 1 + 1 = 2
num_vec_workspaces(::FreeEnergyDistance, ::RSP) = 2
num_vec_workspaces(::PowerMeanProximity, ::RSP) = 2

function compute_target(::Distance, ti::TargetInit{<:Euclidean})
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2), (b1, b2)) = hypot((b1 - a1), (b2 - a2))

    targetcol = vec_workspace(ti) .= _hypot.(sourceids(ti), (targetspatialidx(ti),))
    return readonlyarray(targetcol)
end
compute_target(::Distance, ti::TargetInit{<:LCP}) = readonlyarray(ti.shortest_paths.dists)

function compute_target(
    ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
)
    (; F_IW) = ti
    node = targetnode(ti)
    # Set target rowsum of PC to zero
    PC_rowsums = vec_workspace(ti) .= ti.PC_rowsums
    PC_rowsums[node] = 0
    # Solve (I - W) \ PC_rowsums
    return readonlyarray(ldiv!(ti, F_IW, PC_rowsums))
end

# RSP
function compute_target(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Y, Zⁱ) = ti
    C̄ = vec_workspace(ti) .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[targetnode(ti)]
    return readonlyarray(C̄)
end
function compute_target(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    (; θ) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    targetcol = vec_workspace(ti) .= -log.(max.(0, sp)) ./ θ
    return readonlyarray(targetcol)
end
function compute_target(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; θ) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    targetcol = vec_workspace(ti) .= sp .^ (1 / θ)
    return readonlyarray(targetcol)
end
function compute_target(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z) = ti
    targetcol = vec_workspace(ti) .= Z ./ Z[targetnode(ti)]
    return readonlyarray(targetcol)
end

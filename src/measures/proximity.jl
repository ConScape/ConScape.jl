# All ProximityMeasure are grouped here as they are relatively trivial

"""
    ProximityMeasure <: GraphMeasure

Abstract supertype for measures that can be used
as proximities (TODO: explain what proximities are)

They return a sparse array from `solve`.
"""
abstract type ProximityMeasure <: GraphMeasure end

# All ProximityMeasure return a SparseArray of single s/t values
returntrait(::ProximityMeasure) = ReturnAssignedSparse()

"""
    DistanceMeasure <: ProximityMeasure

Abstract supertype for measures that can be used as proximities,
but first need conversion with a `distance_transformation`.

They return a sparse array from `solve`.
"""
abstract type DistanceMeasure <: ProximityMeasure end

# TODO: docs
struct Distance <: DistanceMeasure end
struct ExpectedCost <: DistanceMeasure end
struct FreeEnergyDistance <: DistanceMeasure end
struct HittingTime <: DistanceMeasure end

# Not conditional upon arrival
struct PowerMeanProximity <: ProximityMeasure end
struct SurvivalProbability <: ProximityMeasure end

function compute_target(::Distance, ti::TargetInit{<:Euclidean})
    _hypot(a::CartesianIndex, b::CartesianIndex) = _hypot(Tuple(a), Tuple(b))
    _hypot((a1, a2)::Tuple, (b1, b2)::Tuple) = hypot((b1 - a1), (b2 - a2))
    return ti.workspace .= _hypot.(sourceids(ti), (targetspatialidx(ti),))
end
compute_target(::Distance, ti::TargetInit{<:LCP}) = readonlyarray(ti.shortest_paths.dists)

function compute_target(
    ::Union{ExpectedCost,FreeEnergyDistance}, ti::TargetInit{<:RandomWalk}
)
    (; IW_factorization) = ti
    node = targetnode(ti)
    PC_rowsums = ti.workspace
    # Set target rowsum of PC to zero
    PC_rowsums .= ti.PC_rowsums
    PC_rowsums[node] = 0
    # Solve (I - W) \ PC_rowsums
    return ldiv!(ti, IW_factorization, PC_rowsums)
end

# RSP
function compute_target(::ExpectedCost, ti::TargetInit{<:RSP})
    (; Y, Zⁱ) = ti
    C̄ = workspace(ti) .= Y .* Zⁱ
    # Subtract the cost at the target from all sources
    C̄ .-= C̄[targetnode(ti)]
    return C̄
end
function compute_target(::FreeEnergyDistance, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    return workspace .= -log.(max.(0, sp)) ./ θ
end
function compute_target(::PowerMeanProximity, ti::TargetInit{<:RSP})
    (; θ, workspace) = ti
    sp = get_or_compute_target!(ti, SurvivalProbability())
    return workspace .= sp .^ (1 / θ)
end
function compute_target(::SurvivalProbability, ti::TargetInit{<:RSP})
    (; Z, workspace) = ti
    return workspace .= Z ./ Z[targetnode(ti)]
end

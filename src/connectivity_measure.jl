
"""
    MovementMode

Abstract supertype for movement modes.

These define the path distribution of all possible paths between source and targets.
"""
abstract type MovementMode end
"""
    ArrivingMovement

MovementMode where the movement is conditional upon arrival at the target,
assuming immortality within the context of movement.
"""
abstract type ArrivingMovement <: MovementMode end
"""
    AbsorbingMovement

MovementMode where individuals may not arrive at the target,
allowing simulation of mortality.
"""
abstract type AbsorbingMovement <: MovementMode end

"""
    RandomisedShortestPath <: ArrivingMovementMode

Randomised shortest path movement. Intermediate between LeastCost and RandomWalk.

Assumes partial knowledge and immortality.

## Keywords

- `θ`: the probability of teleportation
- `diagvalue`: the value to use for the diagonal of the proximity matrix
- `approx`: whether to use an approximate algorithm
"""
struct RandomisedShortestPath{CM<:FundamentalMeasure,T<:Union{Real,Nothing},DV} <: ArrivingMovement
    connectivity_maesure::CM
    θ::T
    diagvalue::DV = nothing
    approx::Bool = false
end
"""
    LeastCost <: ArrivingMovementMode

Identical to RSP with theta of Inf (if that could run).

Assumes infinite knowledge and immortality.
"""
struct LeastCost <: ArrivingMovement end
"""
    RandomWalk <: ArrivingMovementMode

Identical to RSP with theta of 0 (if that could run).

Assumes zero knowledge but immortality.
"""
struct RandomWalk <: ArrivingMovement end
"""
    AbsorbingRandomWalk <: AbsorbingMovementMode

Similar to RSP with theta of 1, but not conditional upon arrival.

Allows integration of mortality with movement.
"""
struct AbsorbingRandomWalk <: AbsorbingMovement end

abstract type Measure end

"""
    SourceTargetMeasure 

Abstract supertype for source-target measures.

These characterize distance, proximity or path distribution between source and target pixels.

These produce a dense fundamental matrix, but may return a summary of it such as the mean.
"""
abstract type SourceTargetMeasure <: Measure end

abstract type PathDistributionMeasure <: SourceTargetMeasure end

abstract type ConnectivityMeasure <: SourceTargetMeasure end
abstract type FundamentalMeasure <: ConnectivityMeasure end
abstract type DistanceMeasure <: FundamentalMeasure end

@kwdef struct ExpectedCost{DT} <: DistanceMeasure
    distance_transformation::DT
end
@kwdef struct FreeEnergyDistance{DT} <: DistanceMeasure
    distance_transformation::DT
end
struct PowerMeanProximity <: FundamentalMeasure end
# TODO: look at theta use for SurvivalProbability, it should be 1
struct SurvivalProbability <: FundamentalMeasure end
struct KullbackLeiblerDivergence <: PathDistributionMeasure end
struct HittingTime <: DistanceMeaure end

keywords(cm::ConnectivityMeasure) = _keywords(cm)

distance_transformation(cm::FundamentalMeasure) = nothing
distance_transformation(cm::DistanceMeasure) = cm.distance_transformation

returntrait(::ConnectivityMeasure) = SumDenseSpatial()
returntrait(::KullbackLeiblerDivergence) = SumScalar()
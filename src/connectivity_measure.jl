
"""
    MovementMode

Abstract supertype for movement modes.

These define the path distribution of all possible paths between source and targets.
"""
abstract type MovementMode end
abstract type ArrivingMovementMode <: MovementMode end
abstract type AbsorbingMovementMode <: MovementMode end

"""
    RandomisedShortestPath <: ArrivingMovementMode

Randomised shortest path movement. Intermediate between LeastCost and RandomWalk.

Assumes partial knowledge and immortality.

## Keywords

- `θ`: the probability of teleportation
- `diagvalue`: the value to use for the diagonal of the proximity matrix
- `approx`: whether to use an approximate algorithm
"""
struct RandomisedShortestPath{T<:Union{Real,Nothing},DV} <: ArrivingMovementMode
    θ::T
    diagvalue::DV = nothing
    approx::Bool = false
end
"""
    LeastCost <: ArrivingMovementMode

Identical to RSP with theta of Inf (if that could run).

Assumes infinite knowledge, and immortality.
"""
struct LeastCost <: ArrivingMovementMode end
"""
    RandomWalk <: ArrivingMovementMode

Identical to RSP with theta of 0 (if that could run).

Assumes zero knowledge, but immortality.
"""
struct RandomWalk <: ArrivingMovementMode end
"""
    AbsorbingRandomWalk <: AbsorbingMovementMode

Similar to RSP with theta of 1, but not conditional upon arrival.

Allows integration of mortality with movement
"""
struct AbsorbingRandomWalk <: AbsorbingMovementMode end

"""
    SourceTargetMeasure 

Abstract supertype for source-target measures.

These characterize distance, proximity or path distribution between source and target pixels.

These produce a dense fundamental matrix, but may return a summary of it such as the mean.
"""
abstract type SourceTargetMeasure end

abstract type PathDistributionMeasure <: SourceTargetMeasure end

abstract type ConnectivityMeasure <: SourceTargetMeasure end
abstract type FundamentalMeasure <: ConnectivityMeasure end
abstract type DistanceMeasure <: FundamentalMeasure end

@kwdef struct ExpectedCost{M<:ArrivingMovementMode,DT} <: DistanceMeasure
    movement::M
    distance_transformation::DT
end
@kwdef struct FreeEnergyDistance{M<:ArrivingMovementMode,DT} <: DistanceMeasure
    movement::M
    distance_transformation::DT
end
@kwdef struct PowerMeanProximity{M<:RSP} <: FundamentalMeasure
    movement::M
end
# TODO: look at theta use for SurvivalProbability, it should be 1
@kwdef struct SurvivalProbability{M<:AbsorbingRandomWalk} <: FundamentalMeasure
    movement::M
end
@kwdef struct KullbackLeiblerDivergence{M<:ArrivingMovementMode,F} <: PathDistributionMeasure 
    movement::M
    summary::F = mean
end

keywords(cm::ConnectivityMeasure) = _keywords(cm)

distance_transformation(cm::FundamentalMeasure) = nothing
distance_transformation(cm::DistanceMeasure) = cm.distance_transformation

# TODO remove the complexity of the connectivity_function
# These methods are mostly to avoid changing the original interface for now
# Its a quirk of how MeanKullbackLeiblerDivergence is implemented
# that these can be calculated separately from the main grid
source_target_function(::LeastCostDistance) = least_cost_distance
source_target_function(::ExpectedCost) = expected_cost
source_target_function(::FreeEnergyDistance) = free_energy_distance
source_target_function(::SurvivalProbability) = survival_probability
source_target_function(::PowerMeanProximity) = power_mean_proximity

source_target_function(m::KullbackLeiblerDivergence{LeastCost,typeof(mean)}) = mean_lc_kl_divergence
source_target_function(m::KullbackLeiblerDivergence{RandomisedShortestPath,typeof(mean)}) = mean_kl_divergence
source_target_function(m::KullbackLeiblerDivergence{RandomWalk,typeof(mean)}) = (x...; kw...) -> 0.0


# This is not used yet but could be
compute(cm::SourceTargetMeasure, g; kw...) =
    connectivity_function(m)(g; keywords(cm)..., kw...)

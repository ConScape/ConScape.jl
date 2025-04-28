"""
    MovementMode

Abstract supertype for movement modes.

These define the path distribution of all possible paths between source and targets.
"""
abstract type MovementMode end

init(movement_mode::MovementMode, grid::Grid; kw...) =
    init(Problem(; movement_mode), grid; kw...)

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

const PROXIMITY_KEYWORDS = """
- `proximity_measure`: the measure to use for the probability of arrival at the target.
    By default this is `ExpectedCost()`
- `distance_transformation`: the transformation to apply to a distance matrix to 
    convert it to a proximity matrix. [`DistanceMeasure`](@ref)s like `ExpectedCost()` and 
    `FreeEnergyDistance()` use this transformation, but `ProximityMeasure`s like 
    `PowerMeanProximity()` or `SurvivalProbability()` do not.
- `diagvalue`: The value to use for the diagonal of the proximity matrix.
"""

"""
    RandomisedShortestPath <: ArrivingMovementMode

Randomised shortest path movement. Intermediate between LeastCost and RandomWalk.

Assumes partial knowledge and immortality.

## Keywords

$PROXIMITY_KEYWORDS
- `theta`: The probability of arrival at the target.
- `approx`: Whether to use an approximate algorithm, `false` by default.
"""
@kwdef struct RandomisedShortestPath{
    PM<:Union{DistanceMeasure,ProximityMeasure,Nothing},DT,T<:Union{Real,Nothing},DV
} <: ArrivingMovement
    proximity_measure::PM = ExpectedCost()
    distance_transformation::DT = nothing
    diagvalue::DV = nothing
    theta::T = nothing
    approx::Bool = false
end
RandomisedShortestPath(proximity_measure; kw...) =
    RandomisedShortestPath(; proximity_measure, kw...)

const RSP = RandomisedShortestPath

proximity_measure(mm::RandomisedShortestPath) = mm.proximity_measure
distance_transformation(mm::RandomisedShortestPath) = mm.distance_transformation
diagvalue(mm::RandomisedShortestPath) = mm.diagvalue
approx(mm::RandomisedShortestPath) = mm.approx
theta(mm::RandomisedShortestPath) = mm.theta

"""
    LeastCost <: ArrivingMovementMode

Identical to RSP with theta of Inf, if that could run without numerical errors.

Assumes infinite knowledge and immortality.

## Keywords

$PROXIMITY_KEYWORDS
"""
struct LeastCost <: ArrivingMovement end

"""
    RandomWalk <: ArrivingMovementMode

Identical to RSP with theta of 0, if that could run without numerical errors.

Performance is usually 2-3 times slower than RSP>

Assumes zero knowledge but immortality.

## Keywords

$PROXIMITY_KEYWORDS
"""
struct RandomWalk <: ArrivingMovement end

"""
    AbsorbingRandomWalk <: AbsorbingMovementMode

Similar to RSP with theta of 1, but not conditional upon arrival.

Allows integration of mortality with movement.
"""
struct AbsorbingRandomWalk <: AbsorbingMovement end
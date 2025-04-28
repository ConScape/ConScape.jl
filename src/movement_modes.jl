"""
    MovementMode

Abstract supertype for movement modes.

These define the path distribution of all possible paths between source and targets.
"""
abstract type MovementMode end

distance_transformation(mm::MovementMode) = mm.distance_transformation
diagvalue(mm::MovementMode) = mm.diagvalue

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
- `distance_transformation`: the transformation to apply to a distance matrix to 
    convert it to a proximity matrix. [`DistanceMeasure`](@ref)s like `ExpectedCost()` and 
    `FreeEnergyDistance()` use this transformation, but `ProximityMeasure`s like 
    `PowerMeanProximity()` or `SurvivalProbability()` do not.
- `diagvalue`: The value to use for the diagonal of the proximity matrix.
    (TODO: explain why its relevent)
"""

"""
    RandomisedShortestPath <: ArrivingMovementMode

    RandomisedShortestPath(; kw...)

Randomised shortest path movement. Intermediate between 
[`LeastCost`](@ref) and [`RandomWalk`](@ref).

Assumes partial knowledge and immortality.

## Keywords

- `proximity_measure`: the measure to use for the probability of arrival at the target.
    By default this is `ExpectedCost()`
$PROXIMITY_KEYWORDS
- `theta`: the inverse temperature (TODO: in more ecological terms)
- `approx`: Whether to use an approximate algorithm, `false` by default.
    (TODO: more detail)
"""
@kwdef struct RandomisedShortestPath{
    PM<:Union{ProximityMeasure,Nothing},DT,T<:Union{Real,Nothing},DV
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
approx(mm::RandomisedShortestPath) = mm.approx
theta(mm::RandomisedShortestPath) = mm.theta

"""
    LeastCost <: ArrivingMovementMode

    LeastCost(; kw...)

Identical to [`RandomisedShortestPath`](@ref) with `theta` of `Inf`, 
if that could run without numerical problems.

Assumes infinite knowledge and immortality.

## Keywords

$PROXIMITY_KEYWORDS
"""
@kwdef struct LeastCost{DT,DV} <: ArrivingMovement 
    distance_transformation::DT = nothing
    diagvalue::DV = nothing
end

proximity_measure(mm::LeastCost) = ExpectedCost()

const LC = LeastCost

"""
    RandomWalk <: ArrivingMovementMode

    RandomWalk(; kw...)


Performance is usually 2-3 times slower than RSP>

Identical to [`RandomisedShortestPath`](@ref) with `theta` of `0`, 
if that could run without numerical problems.

Assumes zero knowledge but immortality.

## Keywords

$PROXIMITY_KEYWORDS
"""
@kwdef struct RandomWalk{DT,DV} <: ArrivingMovement 
    distance_transformation::DT = nothing
    diagvalue::DV = nothing
end

proximity_measure(mm::RandomWalk) = ExpectedCost()
theta(mm::RandomWalk) = 0

const RW = RandomWalk

"""
    AbsorbingRandomWalk <: AbsorbingMovementMode

Similar to RSP with theta of 1, but not conditional upon arrival.

Allows integration of mortality with movement.
"""
struct AbsorbingRandomWalk <: AbsorbingMovement end
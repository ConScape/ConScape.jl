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

"""
    RandomisedShortestPath <: ArrivingMovementMode

Randomised shortest path movement. Intermediate between LeastCost and RandomWalk.

Assumes partial knowledge and immortality.

## Keywords

- `θ`: the probability of teleportation
- `diagvalue`: the value to use for the diagonal of the proximity matrix
- `approx`: whether to use an approximate algorithm
"""
@kwdef struct RandomisedShortestPath{
    PM<:Union{DistanceMeasure,ProximityMeasure,Nothing},DT,T<:Union{Real,Nothing},DV
} <: ArrivingMovement
    proximity_measure::PM = ExpectedCost()
    distance_transformation::DT = nothing
    theta::T = nothing
    diagvalue::DV = nothing
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
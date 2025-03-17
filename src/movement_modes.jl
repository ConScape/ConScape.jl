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
@kwdef struct RandomisedShortestPath{CM<:FundamentalMeasure,T<:Union{Real,Nothing},DV} <: ArrivingMovement
    connectivity_measure::CM
    theta::T
    diagvalue::DV = nothing
    approx::Bool = false
end
RandomisedShortestPath(connectivity_measure::FundamentalMeasure; kw...) =
    RandomisedShortestPath(; connectivity_measure, kw...)

connectivity_measure(mm::RandomisedShortestPath) = mm.connectivity_measure
distance_transformation(mm::RandomisedShortestPath) = 
    distance_transformation(connectivity_measure(mm))
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
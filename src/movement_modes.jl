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

"""
    RandomisedShortestPath <: ArrivingMovementMode

    RandomisedShortestPath(; theta, distance_transformation, diagvalue, approx)

Randomised shortest path movement. Intermediate between 
[`LeastCost`](@ref) and [`RandomWalk`](@ref).

Assumes partial knowledge and immortality.

## Keywords

- `theta`: the inverse temperature (TODO: in more ecological terms)
- `diagvalue`: the value to use for the diagonal of the proximity matrix.
    (TODO: explain why its relevent)
- `approx`: whether to use an approximate algorithm
    (TODO: more detail)
"""
@kwdef struct RandomisedShortestPath{
    PM<:Union{ProximityMeasure,Nothing},DT,T<:Union{Real,Nothing},DV
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
approx(mm::RandomisedShortestPath) = mm.approx
theta(mm::RandomisedShortestPath) = mm.theta

"""
    LeastCost <: ArrivingMovementMode

    LeastCost(; distance_transformation, diagvalue)

Identical to [`RandomisedShortestPath`](@ref) with `theta` of `Inf`, 
if that would run.

Assumes infinite knowledge and immortality.
"""
@kwdef struct LeastCost{DT,DV} <: ArrivingMovement 
    distance_transformation::DT = nothing
    diagvalue::DV = nothing
end

proximity_measure(mm::LeastCost) = ExpectedCost()

const LC = LeastCost

"""
    RandomWalk <: ArrivingMovementMode

    RandomWalk(; distance_transformation, diagvalue)

Identical to [`RandomisedShortestPath`](@ref) with `theta` of `0`, 
if that could run.

Assumes zero knowledge but immortality.
"""
@kwdef struct RandomWalk{DT,DV} <: ArrivingMovement 
    distance_transformation::DT = nothing
    diagvalue::DV = nothing
end

proximity_measure(mm::RandomWalk) = ExpectedCost()

const RW = RandomWalk

"""
    AbsorbingRandomWalk <: AbsorbingMovementMode

Similar to RSP with theta of 1, but not conditional upon arrival.

Allows integration of mortality with movement.
"""
struct AbsorbingRandomWalk <: AbsorbingMovement end
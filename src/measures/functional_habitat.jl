"""
    FunctionalHabitat <: SpatialMeasure

    FunctionalHabitat()

Compute connected habitat of all sources weighted by qualities of 
source s and target t and the proximity between s and t, 
as defined by the [`MovementMode`](@ref)).

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
struct FunctionalHabitat <: SpatialMeasure end

computelevel(::FunctionalHabitat) = TargetLevel()
returntrait(::FunctionalHabitat) = ReturnSpatialTargetSum()

# Its just M
compute_target(::FunctionalHabitat, ti::TargetInit{<:Union{RSP,RandomWalk,LCP}}) = ti.M

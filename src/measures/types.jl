"""
    Measure

Abstract supertype for all ConScape.jl measures.
"""
abstract type Measure end

const MeasureTuple = Tuple{<:Measure,Vararg{Measure}} where N
const MeasureNamedTuple = NamedTuple{K,<:MeasureTuple} where K
const MeasureTupleOrNamedTuple = Union{MeasureTuple,MeasureNamedTuple}

"""
    GraphMeasure <: Measure

Measure in graph space, that return a sparse array from `solve`.
"""
abstract type GraphMeasure <: Measure end

"""
    SpatialMeasure <: Measure

Measure in physical space, that return a dense raster matrix from `solve`.
"""
abstract type SpatialMeasure <: Measure end

# Spatial measures usually return the sum over all 
# target nodes as a spatial raster.
returntrait(::SpatialMeasure) = ReturnSpatialTargetSum()

"""
    PathDistributionMeasure <: Measure

Measures of path distribution, that return a scalar from `solve`.
"""
abstract type PathDistributionMeasure <: Measure end

"""
    ProximityMeasure <: GraphMeasure

Abstract supertype for measures that can be used
as proximities (TODO: explain what proximities are)

They return a sparse array from `solve`.
"""
abstract type ProximityMeasure <: GraphMeasure end

"""
    DistanceMeasure <: ProximityMeasure

Abstract supertype for measures that can be used as proximities,
but first need conversion with a `distance_transformation`.

They return a sparse array from `solve`.
"""
abstract type DistanceMeasure <: ProximityMeasure end

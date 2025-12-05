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

returntrait(::PathDistributionMeasure) = ReturnScalarSum()

# Workspace allocation traits 
# TODO: make these accurate
needs_workspaces(::Measure) = 2
# Count how many workspaces are needed for a problem
nworkspaces(p::AbstractProblem) =
    isempty(measures(p)) ? 0 : mapreduce(needs_workspaces, +, measures(p))

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)

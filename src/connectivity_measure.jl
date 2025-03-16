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
struct HittingTime <: DistanceMeasure end

distance_transformation(cm::FundamentalMeasure) = nothing
distance_transformation(cm::DistanceMeasure) = cm.distance_transformation

returntrait(::ConnectivityMeasure) = SumDenseSpatial()
returntrait(::KullbackLeiblerDivergence) = SumScalar()
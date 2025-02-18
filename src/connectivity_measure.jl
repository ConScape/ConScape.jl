"""
    GraphMeasure 

Abstract supertype for connectivity measures.

These are lazy definitions of conscape functions,
with required parameters attached rather than passed 
in through keywords.
"""
abstract type ConnectivityMeasure end

abstract type FundamentalMeasure <: ConnectivityMeasure end
abstract type DistanceMeasure <: FundamentalMeasure end

struct LeastCostDistance <: ConnectivityMeasure end
@kwdef struct ExpectedCost{T<:Union{Real,Nothing},CM,DV} <: DistanceMeasure
    θ::T
    distance_transformation::CM
    diagvalue::DV = nothing
    approx::Bool = false
end
@kwdef struct FreeEnergyDistance{T<:Union{Real,Nothing},CM,DV} <: DistanceMeasure
    θ::T
    distance_transformation::CM
    diagvalue::DV = nothing
    approx::Bool = false
end
@kwdef struct SurvivalProbability{T<:Union{Real,Nothing},DV} <: FundamentalMeasure
    θ::T = nothing
    diagvalue::DV = nothing # TODO should be 1
    approx::Bool = false
end
@kwdef struct PowerMeanProximity{T<:Union{Real,Nothing},DV} <: FundamentalMeasure
    θ::T = nothing
    diagvalue::DV = nothing
    approx::Bool = false
end

keywords(cm::ConnectivityMeasure) = _keywords(cm)

distance_transformation(cm::FundamentalMeasure) = nothing
distance_transformation(cm::DistanceMeasure) = cm.distance_transformation

# TODO remove the complexity of the connectivity_function
# These methods are mostly to avoid changing the original interface for now
# Its a quirk of how MeanKullbackLeiblerDivergence is implemented
# that these can be calculated separately from the main grid
connectivity_function(::LeastCostDistance) = least_cost_distance
connectivity_function(::ExpectedCost) = expected_cost
connectivity_function(::FreeEnergyDistance) = free_energy_distance
connectivity_function(::SurvivalProbability) = survival_probability
connectivity_function(::PowerMeanProximity) = power_mean_proximity

# This is not used yet but could be
compute(cm::ConnectivityMeasure, g; kw...) =
    connectivity_function(m)(g; keywords(cm)..., kw...)

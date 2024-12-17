# New type-based interface
# Easier to add parameters to these
abstract type ConnectivityMeasure end

abstract type FundamentalMeasure <: ConnectivityMeasure end
abstract type RSPDistanceMeasure <: FundamentalMeasure end

struct LeastCostDistance <: ConnectivityMeasure end
@kwdef struct ExpectedCost{T<:Union{Real,Nothing}} <: RSPDistanceMeasure 
    θ::T=nothing
    approx::Bool=false
end
@kwdef struct FreeEnergyDistance{T<:Union{Real,Nothing}} <: RSPDistanceMeasure 
    θ::T=nothing
    approx::Bool=false
end
@kwdef struct SurvivalProbability{T<:Union{Real,Nothing}} <: FundamentalMeasure 
    θ::T=nothing
    approx::Bool=false
end
@kwdef struct PowerMeanProximity{T<:Union{Real,Nothing}} <: FundamentalMeasure 
    θ::T=nothing
    approx::Bool=false
end

keywords(cm::ConnectivityMeasure) = _keywords(cm)

# TODO remove the complexity of the connectivity_function
# These methods are mostly to avoid changing the original interface for now
connectivity_function(::LeastCostDistance) = least_cost_distance
connectivity_function(::ExpectedCost) = expected_cost
connectivity_function(::FreeEnergyDistance) = free_energy_distance
connectivity_function(::SurvivalProbability) = survival_probability
connectivity_function(::PowerMeanProximity) = power_mean_proximity

# This is not used yet but could be
compute(cm::ConnectivityMeasure, g) = 
    connectivity_function(m)(g; keywords(cm)...)
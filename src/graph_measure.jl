abstract type GraphMeasure end

keywords(o::GraphMeasure) = _keywords(o)

abstract type TopologicalMeasure <: GraphMeasure end
abstract type BetweennessMeasure <: GraphMeasure end
abstract type PerturbationMeasure <: GraphMeasure end
abstract type PathDistributionMeasure <: GraphMeasure end

struct BetweennessQweighted <: BetweennessMeasure end
@kwdef struct BetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end
struct EdgeBetweennessQweighted <: BetweennessMeasure end
@kwdef struct EdgeBetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end

@kwdef struct ConnectedHabitat{DV} <: GraphMeasure
    diagvalue::DV=nothing
end

@kwdef struct Criticality{DV,AV,QT,QS} <: PerturbationMeasure 
    diagvalue::DV=nothing
    avalue::AV=floatmin()
    qˢvalue::QS=0.0
    qᵗvalue::QT=0.0
end

# These maybe don't quite belong here
@kwdef struct EigMax{F,DV,T} <: TopologicalMeasure
    diagvalue::DV=nothing
    tol::T=1e-14
end
struct MeanLeastCostKullbackLeiblerDivergence <: PathDistributionMeasure end
struct MeanKullbackLeiblerDivergence <: PathDistributionMeasure end

# Kind of a hack for now but makes the input requirements clear
keywords(o::GraphMeasure, dt, p::AbstractProblem) = 
    (; _keywords(o)..., distance_transformation=dt)
keywords(o::Union{BetweennessQweighted,EdgeBetweennessQweighted,PathDistributionMeasure}, dt, p::AbstractProblem) = 
    (; _keywords(o)...)
keywords(o::Union{BetweennessKweighted,EigMax}, dt, p::AbstractProblem) = 
    (; _keywords(o)..., 
        connectivity_function=connectivity_function(p), 
        distance_transformation=dt)
keywords(o::ConnectedHabitat, dt, p::AbstractProblem) = 
    (; _keywords(o)..., 
        distance_transformation=dt,
        connectivity_function=connectivity_function(p), 
        approx=connectivity_measure(p).approx) 

graph_function(m::BetweennessKweighted) = betweenness_kweighted
graph_function(m::EdgeBetweennessKweighted) = edge_betweenness_kweighted
graph_function(m::BetweennessQweighted) = betweenness_qweighted
graph_function(m::EdgeBetweennessQweighted) = edge_betweenness_qweighted
graph_function(m::ConnectedHabitat) = connected_habitat
graph_function(m::Criticality) = criticality
graph_function(m::EigMax) = eigmax  
graph_function(m::MeanLeastCostKullbackLeiblerDivergence) = mean_lc_kl_divergence
graph_function(m::MeanKullbackLeiblerDivergence) = mean_kl_divergence

compute(m::GraphMeasure, p::AbstractProblem, g::Union{Grid,GridRSP}) = 
    compute(m, p.distance_transformation, p, g)
compute(m::GraphMeasure, dts::Union{Tuple,NamedTuple}, p::AbstractProblem, g::Union{Grid,GridRSP}) = 
    map(dts) do dt
        compute(m, dt, p, g)
    end
compute(m::GraphMeasure, distance_transformation, p::AbstractProblem, g::Union{Grid,GridRSP}) = 
    graph_function(m)(g; keywords(m, distance_transformation, p)...)
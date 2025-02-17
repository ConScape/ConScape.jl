"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure end

abstract type TopologicalMeasure <: GraphMeasure end
abstract type BetweennessMeasure <: GraphMeasure end
abstract type PerturbationMeasure <: GraphMeasure end
abstract type PathDistributionMeasure <: GraphMeasure end

# Concrete GraphMeasure structs

struct BetweennessQweighted <: BetweennessMeasure end
@kwdef struct BetweennessKweighted <: BetweennessMeasure end
struct EdgeBetweennessQweighted <: BetweennessMeasure end
@kwdef struct EdgeBetweennessKweighted <: BetweennessMeasure end

@kwdef struct ConnectedHabitat <: GraphMeasure end

@kwdef struct Criticality{AV,QT,QS} <: PerturbationMeasure
    avalue::AV = floatmin()
    qˢvalue::QS = 0.0
    qᵗvalue::QT = 0.0
end

@kwdef struct EigMax{T} <: TopologicalMeasure
    tol::T = 1e-14
end

struct MeanLeastCostKullbackLeiblerDivergence <: PathDistributionMeasure end
struct MeanKullbackLeiblerDivergence <: PathDistributionMeasure end

# Map structs to function calls

graph_function(m::BetweennessKweighted) = betweenness_kweighted
graph_function(m::BetweennessQweighted) = betweenness_qweighted
graph_function(m::ConnectedHabitat) = connected_habitat
graph_function(m::Criticality) = criticality
graph_function(m::MeanLeastCostKullbackLeiblerDivergence) = mean_lc_kl_divergence
graph_function(m::MeanKullbackLeiblerDivergence) = mean_kl_divergence
graph_function(m::EdgeBetweennessKweighted) = edge_betweenness_kweighted
graph_function(m::EdgeBetweennessQweighted) = edge_betweenness_qweighted
graph_function(m::EigMax) = eigmax

# Function keywords

keywords(gm::GraphMeasure, p::AbstractProblem) =
    (; _keywords(gm)..., solver=solver(p), _connectivity_keywords(gm, p)...)
keywords(gm::ConnectedHabitat, p::AbstractProblem) =
    (; _keywords(gm)..., approx=connectivity_measure(p).approx, solver=solver(p), _connectivity_keywords(gm, p)...)
function _connectivity_keywords(gm::GraphMeasure, p::AbstractProblem)
    cm = connectivity_measure(p)
    if needs_connectivity(gm)
        (;
            _keywords(gm)...,
            distance_transformation=distance_transformation(cm),
            connectivity_function=connectivity_function(cm)
        )
    else
        _keywords(gm)
    end
end

# Traits

abstract type ReturnType end
struct ReturnsDenseSpatial <: ReturnType end
struct ReturnsSparse <: ReturnType end
struct ReturnsScalar <: ReturnType end
struct ReturnsOther{F} <: ReturnType
    f::F
end

# These allow calculation of return allocations
returntype(::EdgeBetweennessQweighted) = ReturnsSparse()
returntype(::EdgeBetweennessKweighted) = ReturnsSparse()
returntype(::BetweennessQweighted) = ReturnsDenseSpatial()
returntype(::BetweennessKweighted) = ReturnsDenseSpatial()
returntype(::ConnectedHabitat) = ReturnsDenseSpatial()
returntype(::Criticality) = ReturnsDenseSpatial()
returntype(::EigMax) = ReturnsOther((n, m) -> n + m)
returntype(::MeanLeastCostKullbackLeiblerDivergence) = ReturnsScalar()
returntype(::MeanKullbackLeiblerDivergence) = ReturnsScalar()

# A trait for connectivity requirement
needs_connectivity(::GraphMeasure) = false
needs_connectivity(::BetweennessKweighted) = true
needs_connectivity(::EdgeBetweennessKweighted) = true
needs_connectivity(::EigMax) = true
needs_connectivity(::ConnectedHabitat) = true
needs_connectivity(::Criticality) = true

# Workspace allocation traits
return_type(::GraphMeasure) = false
needs_inv(::GraphMeasure) = false
needs_inv(::BetweennessMeasure) = true
needs_Z(::GraphMeasure) = true
needs_workspaces(::GraphMeasure) = 0
needs_workspaces(::BetweennessMeasure) = 1
needs_workspaces(::EdgeBetweennessKweighted) = 2
needs_workspaces(::EdgeBetweennessQweighted) = 3
needs_permuted_workspaces(::GraphMeasure) = 0
needs_permuted_workspaces(::EdgeBetweennessKweighted) = 1
needs_proximity(::GraphMeasure) = false
needs_proximity(::Union{BetweennessKweighted,EdgeBetweennessKweighted}) = true
needs_expected_cost(::GraphMeasure) = false
needs_expected_cost(::EdgeBetweennessKweighted) = true
needs_expected_cost(::MeanKullbackLeiblerDivergence) = true
needs_free_energy_distance(::GraphMeasure) = false
needs_free_energy_distance(::MeanKullbackLeiblerDivergence) = true
needs_Aaj_init(::GraphMeasure) = true # TODO which dont?

# Trait helpers

function count_workspaces(p::AbstractProblem)
    gms = graph_measures(p)
    n = mapreduce(needs_workspaces, max, gms)
    if hastrait(needs_expected_cost, gms) || connectivity_function(p) == ConScape.expected_cost
        max(n, 2)
    end
end
count_permuted_workspaces(p::AbstractProblem) =
    mapreduce(needs_permuted_workspaces, max, graph_measures(p))

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)


# compute: run a graph function with the appropriate keywords
compute(gm::GraphMeasure, p::AbstractProblem, g::Union{Grid,GridRSP}; kw...) =
    graph_function(gm)(g; keywords(gm, p)..., kw...)
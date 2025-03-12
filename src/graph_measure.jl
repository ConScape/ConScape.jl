"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure end

abstract type SpatialMeasure <: GraphMeasure end
abstract type TopologicalMeasure <: GraphMeasure end
abstract type BetweennessMeasure <: SpatialMeasure end
abstract type PerturbationMeasure <: SpatialMeasure end

# Betweenness

abstract type BetweennessWeight end

struct Unweighted <: BetweennessWeighting end
struct QualityWeighted <: BetweennessWeighting end
struct ProximityWeighted <: BetweennessWeighting end
struct QualityAndProximityWeighted <: BetweennessWeighting end

@kwdef struct Betweenness{W} <: BetweennessMeasure 
    weighting::W
end
@kwdef struct EdgeBetweenness{W} <: BetweennessMeasure 
    weighting::W
end

weighting(gm::BetweennessMeasure) = gm.weighting


# Sensitivity

abstract type SensitivityContext end

struct Affinity <: SensitivityContext end
struct Cost <: SensitivityContext end
struct Quality <: SensitivityContext end
abstract type CostAndAffinitySensitivityContext end
struct CostAndAffinity <: CostAndAffinitySensitivityContext end
struct AffinityAndCost <: CostAndAffinitySensitivityContext end

abstract type LandscapeMeasure end

struct LandscapeSum <: LandscapeMeasure end
struct LandscapeEigen <: LandscapeMeasure end

@kwdef struct Sensitivity{C<:SeensitivityContext,LM<:LandscapeMeasure} <: PerturbationMeasure
    wrt::C
    landscare_measure::LM
    unitless::Bool
end

# Others

struct ConnectedHabitat <: SpatialMeasure end
@kwdef struct Criticality{AV,QT,QS} <: PerturbationMeasure
    avalue::AV = floatmin()
    qˢvalue::QS = 0.0
    qᵗvalue::QT = 0.0
end

@kwdef struct EigMax{T} <: TopologicalMeasure
    tol::T = 1e-14
end

# Map structs to function calls

graph_function(::Betweenness{QualityAndProximityWeighted}) = betweenness_kweighted
graph_function(::Betweenness{QualityWeighted}) = betweenness_qweighted
graph_function(::ConnectedHabitat) = connected_habitat
graph_function(::Criticality) = criticality
graph_function(::EdgeBetweenness{QualityAndProximityWeighted}) = edge_betweenness_kweighted
graph_function(::EdgeBetweenness{QualityWeighted}) = edge_betweenness_qweighted
graph_function(::EigMax) = eigmax

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

"""
    ReturnTrait

Traits for preallocated return values of GraphMeasures.
"""
abstract type ReturnTrait end
struct ReturnsDenseSpatial <: ReturnTrait end
struct ReturnsSparse <: ReturnTrait end
struct ReturnsScalar <: ReturnTrait end
struct ReturnsOther{F} <: ReturnTrait
    f::F
end

# These allow calculation of return allocations
returntrait(::SpatialMeasure) = ReturnsDenseSpatial()
returntrait(::EdgeBetweenness) = ReturnsSparse()
returntrait(::PathDistributionMeasure) = ReturnsScalar()
returntrait(::EigMax) = ReturnsOther((n, m) -> n + m)

# A trait for connectivity requirement
needs_connectivity(::GraphMeasure) = false
needs_connectivity(::Betweenness{ProximitWeighted}) = true
needs_connectivity(::EdgeBetweenness{ProximitWeighted}) = true
needs_connectivity(::EigMax) = true
needs_connectivity(::ConnectedHabitat) = true
needs_connectivity(::Criticality) = true

# Workspace allocation traits
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
needs_adjoint_init(::GraphMeasure) = true # TODO which dont?

# Graph measure helpers

# Count how many workspaces are needed for a problem
function count_workspaces(p::AbstractProblem)
    gms = graph_measures(p)
    n = mapreduce(needs_workspaces, max, gms)
    if hastrait(needs_expected_cost, gms) || connectivity_function(p) == ConScape.expected_cost
        max(n, 2)
    end
end
count_permuted_workspaces(p::AbstractProblem) =
    mapreduce(needs_permuted_workspaces, max, graph_measures(p))

# Preallocate the output for a graph measure, where needed
allocate_output(gm::GraphMeasure, g::Grid) = 
    allocate_output(returntrait(gm), gm::GraphMeasure, g)
function allocate_output(::ReturnsDenseSpatial, gm::GraphMeasure, g::Grid)
    A = fill(NaN, size(grid))
    A[grid.id_to_grid_coordinate_list] .= 0.0
    return A
end
allocate_output(::ReturnTrait, gm::GraphMeasure, g::Grid) = nothing

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)

# compute: run a graph function with the appropriate keywords
compute(gm::GraphMeasure, p::AbstractProblem, g::Union{Grid,GridRSP}; kw...) =
    graph_function(gm)(g; keywords(gm, p)..., kw...)

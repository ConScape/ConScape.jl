
abstract type Measure end

"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure <: Measure end

abstract type SpatialMeasure <: GraphMeasure end
abstract type PerturbationMeasure <: SpatialMeasure end

# Betweenness

"""
    BetweennessMeasure 

Measures of node and edge betweenness.
"""
abstract type BetweennessMeasure{W} <: SpatialMeasure end

abstract type BetweennessWeighting end

struct Unweighted <: BetweennessWeighting end
struct QualityWeighted <: BetweennessWeighting end
struct ProximityWeighted <: BetweennessWeighting end
struct QualityAndProximityWeighted <: BetweennessWeighting end

@kwdef struct Betweenness{W} <: BetweennessMeasure{W}
    weighting::W
end
@kwdef struct EdgeBetweenness{W} <: BetweennessMeasure{W}
    weighting::W
end

weighting(gm::BetweennessMeasure) = gm.weighting


# Sensitivity

abstract type SensitivityContext end # "With regards to"

struct Affinity <: SensitivityContext end
struct Cost <: SensitivityContext end
struct Quality <: SensitivityContext end
abstract type CostAndAffinitySensitivityContext end
struct CostAndAffinity <: CostAndAffinitySensitivityContext end
struct AffinityAndCost <: CostAndAffinitySensitivityContext end

abstract type LandscapeMeasure end

struct LandscapeSum <: LandscapeMeasure end
struct LandscapeEigen <: LandscapeMeasure end

@kwdef struct Sensitivity{C<:SensitivityContext,LM<:LandscapeMeasure} <: PerturbationMeasure
    context::C
    landscape_measure::LM
    unitless::Bool
end

# Others

struct ConnectedHabitat <: SpatialMeasure end
@kwdef struct Criticality{AV,QT,QS} <: PerturbationMeasure
    avalue::AV = floatmin()
    qˢvalue::QS = 0.0
    qᵗvalue::QT = 0.0
end

@kwdef struct EigMax{T} <: GraphMeasure
    tol::T = 1e-14
end

# # Workspace allocation traits
# needs_inv(::GraphMeasure) = false
# needs_inv(::BetweennessMeasure) = true
needs_workspaces(::Measure) = 0
needs_workspaces(::BetweennessMeasure) = 1
needs_workspaces(::EdgeBetweenness{QualityAndProximityWeighted}) = 2
needs_workspaces(::EdgeBetweenness{QualityWeighted}) = 3
needs_proximity(::Measure) = false
needs_proximity(::BetweennessMeasure{QualityAndProximityWeighted}) = true
needs_expected_cost(::Measure) = false
needs_expected_cost(::EdgeBetweenness{QualityAndProximityWeighted}) = true
# needs_expected_cost(::KullbackLeiblerDivergence) = true
# needs_free_energy_distance(::GraphMeasure) = false
# needs_free_energy_distance(::MeanKullbackLeiblerDivergence) = true
# needs_adjoint_init(::GraphMeasure) = true # TODO which dont?

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)

# Graph measure helpers

# Count how many workspaces are needed for a problem
function nworkspaces(p::AbstractProblem)
    gms = graph_measures(p)
    n = mapreduce(needs_workspaces, max, gms)
    if hastrait(needs_expected_cost, gms) || connectivity_measure(p) isa ExpectedCost
        max(n, 2)
    end
end
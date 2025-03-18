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

@kwdef struct Sensitivity{W<:SensitivityContext,LM<:LandscapeMeasure} <: PerturbationMeasure
    with_regards_to::W
    landscape_measure::LM = LandscapeSum()
    unitless::Bool = false
end

wrt(gm::Sensitivity) = gm.with_regards_to
unitless(gm::Sensitivity) = gm.unitless
landscape_measure(gm::Sensitivity) = gm.landscape_measure

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

# Return type traits
# returntrait(::SpatialMeasure) = AssignDenseSpatial()
returntrait(::ConnectedHabitat) = SumDenseSpatial()
returntrait(::Betweenness) = SumDenseSpatial()
returntrait(::EdgeBetweenness) = AssignSparse()


# Workspace allocation traits
needs_workspaces(::Measure) = 1
needs_workspaces(::BetweennessMeasure) = 2
needs_workspaces(::EdgeBetweenness{QualityAndProximityWeighted}) = 3
needs_workspaces(::EdgeBetweenness{QualityWeighted}) = 4

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)

# Count how many workspaces are needed for a problem
nworkspaces(p::AbstractProblem) = mapreduce(needs_workspaces, +, graph_measures(p))
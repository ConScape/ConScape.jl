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

# Distances and proximities

struct Distance <: DistanceMeasure end
struct ExpectedCost <: DistanceMeasure end
struct FreeEnergyDistance <: DistanceMeasure end
struct HittingTime <: DistanceMeasure end
# Not conditional upon arrival
struct PowerMeanProximity <: ProximityMeasure end
struct SurvivalProbability <: ProximityMeasure end

struct KullbackLeiblerDivergence <: PathDistributionMeasure end

# Betweenness

abstract type BetweennessWeighting end

struct Unweighted <: BetweennessWeighting end

"""
    QualityWeighted <: BetweennessWeighting

    QualityWeighted()

Compute betweenness of nodes or edges weighted by source and target qualities.
"""
struct QualityWeighted <: BetweennessWeighting end
"""
    ProximityWeighted <: BetweennessWeighting

    ProximityWeighted()

Compute betweenness of nodes or edges weighted by the 
proxmimity between source qualities s and target qualities t.
"""
struct ProximityWeighted <: BetweennessWeighting end
"""
    QualityAndProximityWeighted <: BetweennessWeighting

    QualityAndProximityWeighted()

Compute betweenness of nodes or edges weighted by source qualities s 
and target qualities t, and the proximity between s and t.
"""
struct QualityAndProximityWeighted <: BetweennessWeighting end
"""
    CustomWeighted <: BetweennessWeighting

    CustomWeighted(weight)

Holds and arbitrary array of custom weights. 

Used internally.
"""
struct CustomWeighted{W} <: BetweennessWeighting 
    weight::W
end

const WEIGHTING_ARGUMENT = """
- `weighting`: a [`BetweennessWeighting`](@ref): `Unweighted()`, `QualityWeighted()` 
    `ProximityWeighted()` or `QualityAndProximityWeighted()`
"""

"""
    Betweenness <: SpatialMeasure
    
    Betweenness(weighting)

Compute betweenness of all edges weighted by qualities of 
source s and target t and the proximity between s and t,
as defined by the [`MovementMode`](@ref)).

## Arguments

$WEIGHTING_ARGUMENT

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
struct Betweenness{W} <: SpatialMeasure
    weighting::W
end
Betweenness{W}() where W = Betweenness(W())
Betweenness(; weighting) = Betweenness(weighting)

const MovementFlow = Betweenness{QualityAndProximityWeighted}

"""
    EdgeBetweenness <: GraphMeasure

    EdgeBetweenness(weighting)

Compute betweenness of all edges weighted by qualities of 
source s and target t and the proximity between s and t. 

$WEIGHTING_ARGUMENT

Returns a sparse matrix where element (i, j) is the betweenness of edge (i, j).
"""
@kwdef struct EdgeBetweenness{W} <: GraphMeasure
    weighting::W
end

weighting(gm::Betweenness) = gm.weighting
weighting(gm::EdgeBetweenness) = gm.weighting

# Sensitivity

abstract type TopologicalMetric end
struct Summation <: TopologicalMetric end
struct EigenAnalisis <: TopologicalMetric end

abstract type SensitivityType end
struct Sensitivity <: SensitivityType end
struct Elasticity <: SensitivityType end

"""
    SensitivityAnalysis <: SpatialMeasure

    SensitivityAnalysis(; wrt, metric, sentitivitytype)

Compute sensitivity of all nodes. 

## Keywords

- `wrt`: Five types of node sensitivity are implemented: `Affinity()`, `Cost()`, 
    `Quality()`, `CostAndAffinity()` and `AffinityAndCost()`.
- `metric`: Two [`TopologicalMetric`](@ref)s are implemented to summarize the landscape matrix 
    either through summation ([`Summation()`](@ref)) or through eigen analysis [`LandscapeEigen()`](@ref). 
    The default is `Eigen()`.
- `type`: The results can be provided either as sensitivity w.r.t. `Sensitivity()`
    or w.r.t. `Elasticity()`, the latter are also known as elasticities. The default is `Sensitivity()`

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
@kwdef struct SensitivityAnalysis{WRT<:InputType,TM<:TopologicalMetric,ST<:SensitivityType} <: SpatialMeasure
    wrt::WRT
    metric::TM = Summation()
    type::ST = Sensitivity()
end

wrt(gm::SensitivityAnalysis) = gm.wrt
metric(gm::SensitivityAnalysis) = gm.metric
sensitivitytype(gm::SensitivityAnalysis) = gm.type

# Others

"""
    FunctionalHabitat <: SpatialMeasure

    FunctionalHabitat()

Compute connected habitat of all sources weighted by qualities of 
source s and target t and the proximity between s and t, 
as defined by the [`MovementMode`](@ref)).

The value returned from `solve` is a spatial `Raster` or `Matrix`.
"""
struct FunctionalHabitat <: SpatialMeasure end

struct LandscapeMatrix <: GraphMeasure end

@kwdef struct Criticality{AV,QT,QS} <: SpatialMeasure
    avalue::AV = floatmin()
    qˢvalue::QS = 0.0
    qᵗvalue::QT = 0.0
end

"""
    EigenSide

Abstract supertype for sides of [`EigMax`](@ref).

These let us skip computation with `NoLeft` or `NoRight`.

Both are computed by default, e.g. `Left()` and `Right` are used.
"""
abstract type EigenSide end
struct Left <: EigenSide end
struct NoLeft <: EigenSide end
struct Right <: EigenSide end
struct NoRight <: EigenSide end

"""
    EigMax <: Measure

    Eigmax(; kw...)

Compute the largest eigenvalue triple (left vector, value, and right vector) 
of the quality-scaled proximities with respect to the distance/proximity measure 
in the [`MovementMode`](@ref).

## Keywords

`left`: whether to calculate left eigenvector. Defaults to `Left()`, but can be `NoLeft()`.
`left`: whether to calculate righ eigenvector. Defaults to `Right()`, but can be `NoRight()`.
`tol`: tolerance, defaults to `1e-14`.

The triple is always returned, but if `NoLeft` or `NoRight` are used the 
values contained in the left/right vecto will be zeros.
"""
@kwdef struct EigMax{L,R,T} <: Measure
    left::L=Left()
    right::R=Right()
    tol::T = 1e-14
end


Base.Symbol(m::Measure) = nameof(typeof(m))
Base.Symbol(m::Union{Betweenness,EdgeBetweenness}) = Symbol(nameof(typeof(m)), :_, nameof(typeof(weighting(m))))
function Base.Symbol(m::SensitivityAnalysis) 
    Symbol(
        nameof(typeof(m)), :_, 
        nameof(typeof(wrt(m))), :_, 
        nameof(typeof(metric(m))), :_, 
        nameof(typeof(sensitivitytype(m)))
    )
end

# Return type traits
returntrait(::SpatialMeasure) = ReturnDenseSpatialSum()
returntrait(::GraphMeasure) = ReturnAssignedSparse()
returntrait(::PathDistributionMeasure) = ReturnScalarSum()
returntrait(::EdgeBetweenness) = ReturnCustom()
returntrait(::SensitivityAnalysis) = ReturnCustom()
returntrait(::EigMax) = ReturnCustom()

# Workspace allocation traits 
# TODO: make these accurate
needs_workspaces(::Measure) = 2
needs_workspaces(::Betweenness) = 2
needs_workspaces(::EdgeBetweenness) = 4
needs_workspaces(::SensitivityAnalysis{<:Quality}) = 2
needs_workspaces(::SensitivityAnalysis{<:Permeability}) = 6
# Count how many workspaces are needed for a problem
nworkspaces(p::AbstractProblem) =
    isempty(measures(p)) ? 0 : mapreduce(needs_workspaces, +, measures(p))

# Trait aggregator
hastrait(t, gms) = reduce(|, map(t, gms); init=false)

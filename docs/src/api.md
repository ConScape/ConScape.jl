# Functions

ConScape.jl extends the CommonSolve.jl library so that
`solve` and `init` are generic methods to initilise and 
solve ConScape.jl [`Problem`](@ref)s.

As the scale of Jobs can be very large, ConScape adds methods
to assess the work required to run a problem, before it is actually run.

We suggest writing these to disk as json files, using JSON3.jl

```@docs
assess
reassess
```

# Objects

## Movement Modes

```@docs
MovementMode
RandomisedShortestPath
RandomWalk
LeastCost
```

## Measures

Abstract type hierarchy:

```@docs
Measure
```

### Distances and proximities

```@docs
ProximityMeasure
PowerMeanProximity
SurvivalProbability
DistanceMeasure
ExpectedCost
FreeEnergyDistance
HittingTime
```

### Path distributions measures

```@docs
PathDistributionMeasure
KullbackLeiblerDivergence
```

### Spatial measures

```@docs
SpatialMeasure
Betweenness
ConnectedHabitat
Criticality
Sensitivity
```

```@docs
EdgeBetweenness
Eigmax
```

# Betweenness weighting

Betweenness and EdgeBetweenness measure have
weighting parameters that modify their behavior.

```@docs
BetweennessWeighting
Unweighted
QualityWeighted
ProximityWeighted
QualityAndProximityWeighted
CustomWeighted
```

# Sensitivity parameters

Context: this is often referred to as "with regards to".

```@docs
SensitivityContext
Permeability
CostAndAffinitySensitivityContext
Affinity
Cost
Quality
CostToAffinity
AffinityToCost
```

Sensitivity summary mode:

```@docs
SensitivitySummary
LandscapeSum
LandscapeEigen
```

Sensitivity change mode:

```@docs
SensitivityChange
UnitChange
ProportionalChange
```

## Solvers

```@docs
Solver
VectorSolver
LinearSolver
```

## Problems

```@docs
AbstractProblem
Problem
WindowedProblem
BatchProblem
```

## Init internals

These objects are returned from `init` at various levels

```@docs
Initialisation
MultiGridInit
GridInit
TargetInit
Grid
```
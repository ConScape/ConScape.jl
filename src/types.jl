
"""
    Solver

Abstract supertype for ConScape solvers.

These essentially determin how sparse systems of linear 
equations are solved, indide `ldiv!` calls.

[`VectorSolver`](@ref) and [`LinearSolver`](@ref) are the two implementations.
"""
abstract type Solver end

"""
    AbstractProblem

Abstract supertype for ConScape problem specifications.
"""
abstract type AbstractProblem end

# Input types

abstract type InputType end
abstract type Permeability <: InputType end

abstract type AbstractQuality <: InputType end
struct Quality <: AbstractQuality end
struct SourceQuality <: AbstractQuality end
struct TargetQuality <: AbstractQuality end

struct StepLikelihood <: Permeability end
struct StepCost <: Permeability end

# For sensitivity analysis
abstract type StepCostAndLikelihood <: Permeability end
struct StepCostToLikelihood <: StepCostAndLikelihood end
struct StepLikelihoodToCost <: StepCostAndLikelihood end

"""
    ReturnTrait

Traits that determing what object is allocated to store
the return values of `compute` on a [`Measures`](@ref).
"""
abstract type ReturnTrait end
abstract type ReturnSpatial <: ReturnTrait end
abstract type ReturnSparseGraph <: ReturnTrait end

struct ReturnSpatialSourceSum <: ReturnSpatial end
struct ReturnSpatialTargetSum <: ReturnSpatial end
struct ReturnSpatialSourceAndTargetSum <: ReturnSpatial end
struct ReturnCustomSparse <: ReturnSparseGraph end
struct ReturnAssignedSparse <: ReturnSparseGraph end
struct ReturnAssignedDense <: ReturnTrait end
struct ReturnScalarSum <: ReturnTrait end
struct ReturnCustom <: ReturnTrait end

"""
    Level

Traits for specifying the level of a computation or output:
`GridGraphLevel` for the whole gridgraph, `ConnectedGraphLevel` for a
connected subgraph, and `TargetLevel` for a single target pixel.
"""
abstract type Level end
struct TargetLevel <: Level end
struct ConnectedGraphLevel <: Level end
struct GridGraphLevel <: Level end

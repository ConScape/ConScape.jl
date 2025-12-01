
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

abstract type AbstractgQuality <: InputType end
struct Quality <: AbstractgQuality end
struct SourceQuality <: AbstractgQuality end
struct TargetQuality <: AbstractgQuality end

struct StepLikelihood <: Permeability end
struct StepCost <: Permeability end

# For sensitivity analysis
abstract type StepCostAndLikelihood <: Permeability end
struct StepCostToLikelihood <: StepCostAndLikelihood end
struct StepLikelihoodToCost <: StepCostAndLikelihood end

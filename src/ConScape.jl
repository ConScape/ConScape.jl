module ConScape

using ArnoldiMethod
using ConstructionBase
using Graphs
using LinearAlgebra
using Rasters
using SimpleWeightedGraphs
using SparseArrays
using Rasters.DimensionalData

import CommonSolve
import CommonSolve: solve, init

export RandomisedShortestPath, LeastCost, RandomWalk

export ExpectedCost, FreeEnergyDistance, SurvivalProbability, PowerMeanProximity, KullbackLeiblerDivergence

export Betweenness, EdgeBetweenness, ConnectedHabitat, Criticality, EigMax, Sensitivity

export QualityWeighted, QualityAndProximityWeighted, ProximityWeighted

export VectorSolver, LinearSolver

export MinusLog, MinusLogAlpha, Inv, OddsFor, OddsAgainst, ExpMinus, ExpMinusAlpha

export solve, init, assess

export WindowedProblem, BatchProblem

"""
    Solver

Abstract supertype for ConScape solvers.
"""
abstract type AbstractProblem end
abstract type Solver end

# Randomized shortest path algorithms
# Grid struct and methods
include("transformations.jl")
# Grid struct and methods
include("grid.jl")
# Utilities
include("utils.jl")
# Problems
include("workspaces.jl")
include("measures.jl")
include("movement_modes.jl")
include("problem.jl")
include("initialisation.jl")
include("return.jl")
include("solvers.jl")
include("compute_measures.jl")
include("windows.jl")
include("assessment.jl")
include("simulations.jl")

end

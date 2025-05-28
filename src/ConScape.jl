module ConScape

using LinearAlgebra
using Rasters
using SparseArrays

import ArnoldiMethod
import ConstructionBase
import Graphs

import GeometryOps as GO
import GeometryOps.GeoInterface as GI
import SortTileRecursiveTree as STR

using Rasters.DimensionalData
using Rasters.Extents

import CommonSolve: solve, init

using WoodburyMatrices: Woodbury
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph


export RandomisedShortestPath, LeastCostPath, RandomWalk, Euclidean, RSP, LCP

export Distance, ExpectedCost, FreeEnergyDistance, SurvivalProbability, PowerMeanProximity, KullbackLeiblerDivergence

export MovementFlow, Betweenness, EdgeBetweenness, FunctionalHabitat, Criticality, EigMax, SensitivityAnalysis

export QualityWeighted, QualityAndProximityWeighted, ProximityWeighted, Unweighted

export Sensitivity, Elasticity

export Cumulative, Eigen

export Quality, Cost, Likelihood, CostToLikelihood, LikelihoodToCost

export TargetWeight, AverageWeight

export VectorSolver, LinearSolver

export MinusLog, MinusLogAlpha, Inv, OddsFor, OddsAgainst, ExpMinus, ExpMinusAlpha

export solve, init, assess, reassess

export ConScapeProblem, WindowedProblem, BatchProblem

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

include("inputs.jl")
include("utils/workspaces.jl")
include("utils/readonlyarray.jl")
include("utils/bellman_ford.jl")
include("transformations.jl")
include("measures.jl")
include("movement.jl")
include("problem.jl")
include("graphs.jl")
include("initialisation.jl")
include("solve.jl")
include("return.jl")
include("solvers.jl")
include("compute.jl")
include("windows.jl")
include("assessment.jl")
include("utils/coarse_graining.jl")
include("utils/utils.jl")
include("utils/simulations.jl")

end

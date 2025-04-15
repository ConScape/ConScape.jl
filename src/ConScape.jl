module ConScape

using ArnoldiMethod
using ConstructionBase
using Graphs
using LinearAlgebra
using Rasters
using SimpleWeightedGraphs
using SparseArrays
using SortTileRecursiveTree

import GeometryOps as GO
import GeometryOps.GeoInterface as GI

using Rasters.DimensionalData
using Rasters.Extents

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

include("transformations.jl")
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
include("utils.jl")
include("simulations.jl")

end

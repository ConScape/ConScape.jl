module ConScape

using ArnoldiMethod
using ConstructionBase
using Graphs
using LinearAlgebra
using Rasters
using SimpleWeightedGraphs
using SparseArrays
using SortTileRecursiveTree
using WoodburyMatrices

import GeometryOps as GO
import GeometryOps.GeoInterface as GI

using Rasters.DimensionalData
using Rasters.Extents

import CommonSolve
import CommonSolve: solve, init

export RandomisedShortestPath, LeastCost, RandomWalk, Euclidean, RSP

export Distance, ExpectedCost, FreeEnergyDistance, EuclideanDistance, SurvivalProbability, PowerMeanProximity, KullbackLeiblerDivergence

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

include("utils/workspaces.jl")
include("utils/readonlyarray.jl")
include("utils/bellman_ford.jl")
include("utils/graph_generation.jl")
include("transformations.jl")
include("measures.jl")
include("movement.jl")
include("problem.jl")
include("initialisation.jl")
include("return.jl")
include("solvers.jl")
include("compute.jl")
include("windows.jl")
include("assessment.jl")
include("utils/coarse_graining.jl")
include("utils/utils.jl")
include("utils/simulations.jl")

end

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

import CommonSolve: solve, solve!, init

using WoodburyMatrices: Woodbury
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph


export RandomisedShortestPath, LeastCostPath, RandomWalk, Euclidean, RSP, LCP

export Distance, ExpectedCost, FreeEnergyDistance, SurvivalProbability, PowerMeanProximity, KullbackLeiblerDivergence

export MovementFlow, Betweenness, EdgeBetweenness, FunctionalHabitat, Criticality, EigMax, SensitivityAnalysis

export QualityWeighted, QualityAndProximityWeighted, ProximityWeighted, Unweighted

export Sensitivity, Elasticity

export Cumulative, Eigen

export Quality, SourceQuality, TargetQuality, StepCost, StepLikelihood, StepCostToLikelihood, StepLikelihoodToCost

export TargetWeight, AverageWeight

export VectorSolver, LinearSolver

export MinusLog, MinusLogAlpha, Inv, OddsFor, OddsAgainst, ExpMinus, ExpMinusAlpha

export ConScapeProblem, WindowedProblem, BatchProblem

export solve, solve!, init, assess, reassess

include("types.jl")

include("utils/workspaces.jl")
include("utils/readonlyarray.jl")
include("utils/bellman_ford.jl")

include("transformations.jl")
include("graphs.jl")
include("measures.jl")
include("movement.jl")
include("problem.jl")
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

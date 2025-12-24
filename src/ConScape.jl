module ConScape

using LinearAlgebra
using SparseArrays

import ArnoldiMethod
import ConstructionBase
import Graphs
import Mmap
import Rasters

import GeometryOps as GO
import GeometryOps.GeoInterface as GI
import SortTileRecursiveTree as STR
import ReadOnlyArrays
import Random
using DispatchDoctor

import Rasters.Extents
import Rasters.DimensionalData

import CommonSolve: solve, solve!, init

using WoodburyMatrices: Woodbury
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph
using ReadOnlyArrays: ReadOnlyArray
using Rasters: AbstractRaster, AbstractRasterStack, RasterStack, rebuild, dims, mosaic
import Rasters: Raster


export RandomisedShortestPath, LeastCostPath, RandomWalk, Euclidean, RSP, LCP

export Distance, ExpectedCost, FreeEnergyDistance, SurvivalProbability, PowerMeanProximity, MeanKullbackLeiblerDivergence

export MovementFlow, Betweenness, EdgeBetweenness, FunctionalHabitat, EigMax, SensitivityAnalysis

export QualityWeighted, QualityAndProximityWeighted, ProximityWeighted, Unweighted

export Sensitivity, Elasticity

export Summation

export Quality, SourceQuality, TargetQuality, StepCost, StepLikelihood, StepCostToLikelihood, StepLikelihoodToCost

export TargetWeight, AverageWeight

export VectorSolver, LinearSolver

export MinusLog, MinusLogAlpha, Inv, OddsFor, OddsAgainst, ExpMinus, ExpMinusAlpha

export ConScapeProblem, WindowedProblem, BatchProblem

export solve, solve!, init, assess, reassess, estimate_memory_for_centersize


include("types.jl")
include("measures/types.jl")

include("utils/workspaces.jl")
include("utils/bellman_ford.jl")
include("utils/sparse.jl")

include("transformations.jl")
include("graphs.jl")
include("movement.jl")
include("problem.jl")
include("initialisation.jl")
include("solvers.jl")
include("measures/shared.jl")
include("measures/proximity.jl")
include("measures/eigmax.jl")
include("measures/betweenness.jl")
include("measures/edge_betweenness.jl")
include("measures/kullback_leibler.jl")
include("measures/landscape_matrix.jl")
include("measures/functional_habitat.jl")
include("measures/sensitivity.jl")
include("precalculation.jl")
include("solve.jl")
include("output.jl")
include("windows.jl")
include("assessment.jl")

include("utils/coarse_graining.jl")
include("utils/utils.jl")
include("utils/simulations.jl")

end

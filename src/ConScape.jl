module ConScape

using ArnoldiMethod
using ConstructionBase
using Graphs
using LinearAlgebra
using LinearSolve
using Rasters
using SimpleWeightedGraphs
using SparseArrays
using Rasters.DimensionalData

import CommonSolve
import CommonSolve: solve, init

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
include("graph_measure.jl")
include("connectivity_measure.jl")
include("movement_modes.jl")
include("problem.jl")
include("grid_precalculations.jl")
include("return.jl")
include("compute_graph_measures.jl")
include("compute_connectivity_measures.jl")
include("solvers.jl")
include("windows.jl")
include("assessment.jl")

end

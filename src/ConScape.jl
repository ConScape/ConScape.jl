module ConScape

    using SparseArrays, LinearAlgebra
    using Graphs, Plots, SimpleWeightedGraphs, ProgressLogging, ArnoldiMethod
    using Rasters
    using LinearSolve
    using Rasters.DimensionalData

    # Old funcion-based interface
    abstract type ConnectivityFunction <: Function end
    abstract type DistanceFunction <: ConnectivityFunction end
    abstract type ProximityFunction <: ConnectivityFunction end

    struct least_cost_distance   <: DistanceFunction end
    struct expected_cost         <: DistanceFunction end
    struct free_energy_distance  <: DistanceFunction end

    struct survival_probability  <: ProximityFunction end
    struct power_mean_proximity  <: ProximityFunction end

    # Need to define before loading files
    abstract type AbstractProblem end

    # Randomized shortest path algorithms
    include("randomizedshortestpath.jl")
    # Grid struct and methods
    include("grid.jl")
    # GridRSP (randomized shortest path) struct and methods
    include("gridrsp.jl")
    # IO
    include("io.jl")
    # Utilities
    include("utils.jl")
    include("graph_measure.jl")
    include("connectivity_measure.jl")
    include("problem.jl")
    include("tiles.jl")
end

module ConScape

    using SparseArrays, LinearAlgebra
    using LightGraphs, Plots, SimpleWeightedGraphs, ProgressLogging, ArnoldiMethod

    abstract type ConnectivityFunction <: Function end
    abstract type DistanceFunction <: ConnectivityFunction end
    abstract type ProximityFunction <: ConnectivityFunction end

    struct least_cost_distance   <: DistanceFunction end
    struct expected_cost         <: DistanceFunction end
    struct free_energy_distance  <: DistanceFunction end

    struct survival_probability  <: ProximityFunction end
    struct power_mean_proximity  <: ProximityFunction end

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

end

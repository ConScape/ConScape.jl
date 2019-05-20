module ConScape

    using SparseArrays, LinearAlgebra
    using LightGraphs, Plots, SimpleWeightedGraphs

    # Randomized shortest path dissimilarities and betweenness
    include("randomizedshortestpath.jl")
    # Grid struct and methods
    include("grid.jl")
    # Habitat struct and methods
    include("habitat.jl")
    # IO
    include("io.jl")
    # Special matrix for efficient inverse
    include("blocktridiagonal.jl")
    # Utilities
    include("utils.jl")

end

using ConScape
using Test
using Aqua
using SafeTestsets

@testset "ConScape.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ConScape)
    end
    @safetestset "Sparse utils" begin include("sparse.jl") end
    @safetestset "Workspaces" begin include("workspaces.jl") end
    @safetestset "Graphs" begin include("graph_generation.jl") end
    @safetestset "Initialisation" begin include("Initialisation.jl") end
    # @safetestset "basics" begin include("basics.jl") end
    @safetestset "Permeable wall sim" begin include("wall.jl") end
    @safetestset "Problems" begin include("problem.jl") end
    @safetestset "Sensitivity" begin include("sensitivity.jl") end
    @safetestset "WindowedProblems and BatchProblems" begin include("windowed.jl") end
end

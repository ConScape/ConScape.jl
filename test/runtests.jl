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
    @safetestset "Initialisation" begin include("initialisation.jl") end
    @safetestset "Precalculation" begin include("precalculation.jl") end
    @safetestset "Permeable wall sim" begin include("wall.jl") end
    @safetestset "Measures" begin include("measures.jl") end
    @safetestset "Comparison with previous version" begin include("comparisons.jl") end
    @safetestset "Sensitivity comparison with (bugfixed) original implementation" begin include("sensitivity/comparison.jl") end
    @safetestset "Sensitivity validation against simulations" begin include("sensitivity/validation.jl") end
    @safetestset "WindowedProblems and BatchProblems" begin include("windowed.jl") end
    @safetestset "Assessment" begin include("assessment.jl") end
end

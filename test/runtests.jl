using ConScape
using Test
using Aqua
using JET
using SafeTestsets

@testset "ConScape.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ConScape)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(ConScape; target_defined_modules = true)
    end
    @safetestset "Workspaces" begin include("workspaces.jl") end
    @safetestset "ReadOnlyArray" begin include("readonlyarray.jl") end
    # @safetestset "basics" begin include("basics.jl") end
    @safetestset "Problems" begin include("problem.jl") end
    @safetestset "WindowedProblems and BatchProblems" begin include("windowed.jl") end
end
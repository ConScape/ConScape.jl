using ConScape, Test
using ConScape: Workspaces

workspaces = Workspaces(2, 4)
@test workspaces isa Workspaces{Vector{Float64}}

@testset "resize! works on all vectors" begin
    @test length(workspaces) == 2
    resize!(workspaces, 5)
    @test length(workspaces) == 5
    @test all(ws -> length(ws) == 5, workspaces.workspaces)
    resize!(workspaces, 2)
    @test length(workspaces) == 2
end

@testset "take! and put!" begin
    workspace = take!(workspaces)
    @test workspace isa Vector{Float64}
    @test length(workspace) == 2
    @test count(workspaces.unused) == 3
    put!(workspaces, workspace)
    @test count(workspaces.unused) == 4
end

@testset "iteration calls take!" begin
    ws1, ws2, ws2, ws4 = workspaces
    @test count(workspaces.unused) == 0
    @test_throws ErrorException ws1, ws2 = workspaces
    ConScape.free!(workspaces)
    @test count(workspaces.unused) == 4
end

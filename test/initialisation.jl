using ConScape, Test, SparseArrays
using Rasters, ArchGDAL

# Load test data (same as problem.jl)
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

θ = 0.1
rsp = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
measures = (; fh=FunctionalHabitat(), betq=Betweenness(QualityWeighted()))
problem = ConScapeProblem(; movement=rsp, measures)

@testset "GridGraphInit" begin
    ggi = init(problem, rast)

    @testset "construction" begin
        @test ggi isa ConScape.GridGraphInit
        @test ConScape.nconnectedgraphs(ggi) >= 1
    end

    @testset "field getters" begin
        @test ConScape.problem(ggi) === problem
        @test ConScape.gridgraph(ggi) isa ConScape.GridGraph
        @test ConScape.connectedgraphs(ggi) isa Vector{<:ConScape.ConnectedGraph}
        @test ConScape.workspaces(ggi) isa ConScape.Workspaces{<:Vector{Float64}}
        @test ConScape.mworkspaces(ggi) isa ConScape.Workspaces{<:Matrix{Float64}}
        @test ConScape.storage(ggi) isa Dict
        @test ConScape.measures_outputs(ggi) isa NamedTuple
    end

    @testset "forwarding methods" begin
        @test ConScape.movement(ggi) === rsp
        @test ConScape.theta(ggi) == θ
        @test ConScape.measures(ggi) == measures
        @test ConScape.solver(ggi) isa ConScape.Solver
        @test ConScape.proximity_measure(ggi) isa ExpectedCost
        @test ConScape.distance_transformation(ggi) isa ExpMinus
    end

    @testset "GridGraph forwarding" begin
        @test ConScape.stepcost(ggi) isa SparseMatrixCSC{Float64,Int64}
        @test ConScape.steplikelihood(ggi) isa SparseMatrixCSC{Float64,Int64}
        @test ConScape.sourcequality(ggi) isa Matrix{Float64}
        @test ConScape.targetquality(ggi) isa Matrix{Float64}
        @test size(ggi) == size(rast)
    end
end

@testset "ConnectedGraphInit" begin
    ggi = init(problem, rast)
    cgi = init(ggi, 1)

    @testset "construction" begin
        @test cgi isa ConScape.ConnectedGraphInit
        @test ConScape.connectedgraphid(cgi) == 1
    end

    @testset "field getters" begin
        @test ConScape.problem(cgi) === problem
        @test ConScape.gridgraph(cgi) === ConScape.gridgraph(ggi)
        @test ConScape.connectedgraph(cgi) isa ConScape.ConnectedGraph
        @test ConScape.workspaces(cgi) isa ConScape.Workspaces{<:Vector{Float64}}
        @test ConScape.mworkspaces(cgi) isa ConScape.Workspaces{<:Matrix{Float64}}
        @test ConScape.storage(cgi) isa Dict
        @test ConScape.precalculation(cgi) isa NamedTuple
    end

    @testset "ConnectedGraph forwarding" begin
        @test ConScape.nsources(cgi) > 0
        @test ConScape.ntargets(cgi) > 0
        @test ConScape.stepcost(cgi) isa SparseMatrixCSC
        @test ConScape.steplikelihood(cgi) isa SparseMatrixCSC
        @test ConScape.sourcequality(cgi) isa ConScape.ReadOnlyArray{Float64,1}
        @test ConScape.targetquality(cgi) isa ConScape.ReadOnlyArray{Float64,1}
        @test ConScape.sourceids(cgi) isa SubArray{CartesianIndex{2}}
        @test ConScape.targetids(cgi) isa Vector{<:ConScape.TargetID}
    end

    @testset "precalculation access via getproperty" begin
        @test cgi.P isa SparseMatrixCSC{Float64,Int64}
        @test cgi.W isa SparseMatrixCSC{Float64,Int64}
        @test cgi.IW isa SparseMatrixCSC{Float64,Int64}
        @test cgi.CW isa SparseMatrixCSC{Float64,Int64}
        @test cgi.qˢ === ConScape.sourcequality(cgi)
        @test cgi.qᵗ === ConScape.targetquality(cgi)
    end
end

@testset "TargetInit" begin
    ggi = init(problem, rast)
    cgi = init(ggi, 1)
    ti = init(cgi, 1)

    @testset "construction from Int" begin
        @test ti isa ConScape.TargetInit
        @test ConScape.target(ti) isa ConScape.TargetID
        @test ConScape.targetnode(ti) === 1
    end

    @testset "construction from CartesianIndex" begin
        target_idx = ConScape.targetids(cgi)[1].spatialidx
        ti2 = init(cgi, target_idx)
        @test ti2 isa ConScape.TargetInit
        @test ConScape.targetspatialidx(ti2) == target_idx
    end

    @testset "field getters" begin
        @test ConScape.connectedgraphinit(ti) === cgi
        @test ConScape.problem(ti) === problem
        @test ConScape.gridgraph(ti) === ConScape.gridgraph(cgi)
        @test ConScape.connectedgraph(ti) === ConScape.connectedgraph(cgi)
        @test ConScape.storage(ti) === ConScape.storage(cgi)
        @test ConScape.workspaces(ti) === ConScape.workspaces(cgi)
    end

    @testset "ConnectedGraph forwarding" begin
        @test ConScape.nsources(ti) === ConScape.nsources(cgi)
        @test ConScape.ntargets(ti) === ConScape.ntargets(cgi)
        @test ConScape.stepcost(ti) === ConScape.stepcost(cgi)
        @test ConScape.steplikelihood(ti) === ConScape.steplikelihood(cgi)
        @test ConScape.sourcequality(ti) === ConScape.sourcequality(cgi)
        @test ConScape.targetquality(ti) === ConScape.targetquality(cgi)
    end

    @testset "precalculation access via getproperty" begin
        @test ti.P === cgi.P
        @test ti.W === cgi.W
        @test ti.qˢ === cgi.qˢ === ConScape.sourcequality(cgi)
        @test ti.qᵗ isa Float64  # Single target quality value
    end

    @testset "lazy dense vector computation" begin
        @test ti.Z isa ConScape.ReadOnlyArray{Float64,1}
        @test length(ti.Z) == ConScape.nsources(cgi)
    end
end

@testset "init function variations" begin
    @testset "init with RasterStack" begin
        ggi = init(problem, rast)
        @test ggi isa ConScape.GridGraphInit
    end

    @testset "init with GridGraph" begin
        gg = ConScape.GridGraph(problem, rast)
        ggi = init(problem, gg)
        @test ggi isa ConScape.GridGraphInit
    end

    @testset "init directly to ConnectedGraphInit" begin
        cgi = init(problem, rast, 1)
        @test cgi isa ConScape.ConnectedGraphInit
    end

    @testset "init directly to TargetInit" begin
        ti = init(problem, rast, 1, 1)
        @test ti isa ConScape.TargetInit
    end

    @testset "init with movement mode shortcut" begin
        ggi = init(rsp, rast)
        @test ggi isa ConScape.GridGraphInit
    end

    @testset "init with measure and movement" begin
        ggi = init(FunctionalHabitat(), rsp, rast)
        @test ggi isa ConScape.GridGraphInit
    end

    @testset "setmeasures via init" begin
        ggi = init(problem, rast)
        ggi2 = init(FunctionalHabitat(), ggi)
        @test ConScape.measures(ggi2) == (; FunctionalHabitat=FunctionalHabitat())
    end
end

@testset "_check_inputs validation" begin
    gg_full = ConScape.GridGraph(problem, rast)

    # RSP requires both stepcost and steplikelihood
    @test ConScape._check_inputs(RSP(; theta=0.1), gg_full) === nothing

    # LCP requires stepcost
    @test ConScape._check_inputs(LCP(), gg_full) === nothing

    # RandomWalk requires steplikelihood
    @test ConScape._check_inputs(RandomWalk(), gg_full) === nothing

    # Euclidean has no requirements
    @test ConScape._check_inputs(Euclidean(), gg_full) === nothing
end

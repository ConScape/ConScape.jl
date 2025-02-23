using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL
using ConScape.LinearSolve

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

θ = 0.1
landscape = "sno_2000"
# The way the ascii is read in is reversed and rotated from what GDAL does
affinities = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "affinities_$landscape.asc")), NaN)); dims=X)
qualities = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "qualities_$landscape.asc")), NaN)); dims=X)
qualities[(affinities .> 0) .& isnan.(qualities)] .= 1e-20
rast = RasterStack((; affinities, qualities, target_qualities=qualities))

affinities_asc = ConScape.readasc(joinpath(datadir, "affinities_$landscape.asc"))[1]
qualities_asc = ConScape.readasc(joinpath(datadir, "qualities_$landscape.asc"))[1]
qualities_asc[(affinities_asc .> 0) .& isnan.(qualities_asc)] .= 1e-20

graph_measures = (;
# betq=ConScape.BetweennessQweighted(),
    betk=ConScape.BetweennessKweighted(),
    ch=ConScape.ConnectedHabitat(),
    # # TODO sens=ConScape.Sensitivity(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
# Set low alpha here so the decay is steep for testing
distance_transformation = x -> exp(-x / 2)
connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)
expected_layers = (:betk, :ch)

solver = ConScape.MatrixSolver()
# solver = ConScape.VectorSolver()
problem = ConScape.Problem(; graph_measures, connectivity_measure, solver)
solve(problem, rast; verbose=true)

@testset "target mosaicing matches original" begin
    windowed_problem = ConScape.WindowedProblem(problem; 
        buffer=10, centersize=5, threaded=false
    )
    @test collect(ConScape._window_ranges(windowed_problem, rast)) == [
        (1:25, 1:25)   (1:25, 6:30)   (1:25, 11:35)   (1:25, 16:40)   (1:25, 21:45)   (1:25, 26:50)   (1:25, 31:55)   (1:25, 36:59)
        (6:30, 1:25)   (6:30, 6:30)   (6:30, 11:35)   (6:30, 16:40)   (6:30, 21:45)   (6:30, 26:50)   (6:30, 31:55)   (6:30, 36:59)
        (11:35, 1:25)  (11:35, 6:30)  (11:35, 11:35)  (11:35, 16:40)  (11:35, 21:45)  (11:35, 26:50)  (11:35, 31:55)  (11:35, 36:59)
        (16:40, 1:25)  (16:40, 6:30)  (16:40, 11:35)  (16:40, 16:40)  (16:40, 21:45)  (16:40, 26:50)  (16:40, 31:55)  (16:40, 36:59) 
        (21:44, 1:25)  (21:44, 6:30)  (21:44, 11:35)  (21:44, 16:40)  (21:44, 21:45)  (21:44, 26:50)  (21:44, 31:55)  (21:44, 36:59) 
    ]
    test_results = ConScape.solve(windowed_problem, rast; test_windows=true)
    inner_targets = copy(rast.target_qualities)
    replace!(inner_targets, NaN => 0.0)
    # Edge targets are lost with windowing
    inner_targets[1:10, :] .= 0
    inner_targets[:, 1:10] .= 0
    inner_targets[end-9:end, :] .= 0
    inner_targets[:, end-9:end] .= 0
    @test parent(inner_targets) == parent(test_results.target_qualities)
end

@testset "windowed results approximate non-windowed" begin
    buffer=15
    windowed_problem = ConScape.WindowedProblem(problem; 
        buffer, centersize=5
    )
    mask!(rast; with=rast)
    rast_inner = ConScape._get_window_with_zeroed_buffer(windowed_problem, rast, axes(rast))
    @time wp_result = ConScape.solve(windowed_problem, rast)
    @time p_result = ConScape.solve(problem, rast_inner)
    p_result
    # plot(p_result)
    # plot(wp_result)
    @test maplayers(p_result, wp_result) do P, WP
        broadcast(P, WP) do p, wp
            isnan(p) && isnan(wp) || isapprox(p, wp; atol=1e-4)
        end |> all
    end |> all
end


# BatchProblem writes files to disk and mosaics to RasterStack
@testset "batch problem matches windowed problem" begin
    solver = ConScape.VectorSolver()
    # Use a higher alpha to catch differences
    distance_transformation = x -> exp(-x / 50)
    connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)
    problem = ConScape.Problem(; graph_measures, connectivity_measure, solver)

    kw = (; buffer=10, centersize=5)
    windowed_problem = ConScape.WindowedProblem(problem; kw...)
    @time workspace = ConScape.init(windowed_problem, rast);
    @time windowed_result = ConScape.solve!(workspace, windowed_problem);

    batch_problem = ConScape.BatchProblem(problem; datapath=tempname(), kw...)
    ConScape.solve(batch_problem, rast)
    batch_result = mosaic(batch_problem; to=rast)
    @test batch_result isa RasterStack

    # BatchProblem can be run as batch jobs for clusters
    # We just need a new path to make sure the result is from a new run
    batch_jobs_problem = ConScape.BatchProblem(problem; 
        datapath=tempname(), kw...
    )
    assessment = ConScape.assess(batch_jobs_problem, rast)
    batch_jobs_problem.centersize
    @test assessment.njobs == 39

    for job in 1:assessment.njobs
        ConScape.solve(batch_jobs_problem, rast, assessment, job)
    end
    batch_jobs_result = mosaic(batch_jobs_problem; to=rast)
    batch_jobs_result.betk


    @testset "reassessment" begin
        # There should be no jobs left
        re1 = ConScape.reassess(batch_jobs_problem, assessment)
        @test re1.njobs == 0
        @test length(re1.indices) == 0

        # Delete three results
        paths = ConScape.batch_paths(batch_jobs_problem, size(assessment))
        rm.(paths[[1, 7, 21]]; recursive=true)
        re2 = ConScape.reassess(batch_jobs_problem, assessment)
        @test re2.njobs == 3
        @test length(re2.indices) == 3
        @test re2.mask[[1, 7, 21]] == [true, true, true]

        # Run the reassessment
        for job in 1:re2.njobs
            ConScape.solve(batch_jobs_problem, rast, re2, job)
        end

        # Again there are no jobs left
        re3 = ConScape.reassess(batch_jobs_problem, assessment)
        @test re3.njobs == 0
        @test length(re3.indices) == 0
        @test count(re3.mask) == 0
    end


    nested_problem = ConScape.BatchProblem(windowed_problem; 
        datapath=tempname(), centersize=(10, 10)
    )
    ConScape.assess(nested_problem, rast)
    ConScape.solve(nested_problem, rast)
    nested_result = mosaic(nested_problem; to=rast)
    @test nested_result isa RasterStack

    nested_jobs_problem = ConScape.BatchProblem(windowed_problem; 
        datapath=tempname(), centersize=(10, 10)
    )
    # Try one
    @time workspace = ConScape.init(nested_jobs_problem, rast)
    @time ConScape.solve!(workspace, nested_jobs_problem, 5)

    assessment = ConScape.assess(nested_jobs_problem, rast);
    for job in 1:assessment.njobs
        ConScape.solve(nested_jobs_problem, rast, job)
    end
    nested_jobs_result = mosaic(nested_jobs_problem; to=rast)

    @testset "nested reassessment" begin
        # There should be no jobs left
        re1 = ConScape.reassess(nested_jobs_problem, assessment)
        @test re1.njobs == 0
        @test length(re1.indices) == 0

        # Delete three results
        paths = ConScape.batch_paths(nested_jobs_problem, size(assessment))
        rm.(paths[[2, 5]]; recursive=true)
        re2 = ConScape.reassess(nested_jobs_problem, assessment)
        @test re2.njobs == 2
        @test length(re2.indices) == 2
        @test re2.mask[[2, 5]] == [true, true]
        re2
        # Run the reassessment
        for job in 1:re2.njobs
            ConScape.solve(nested_jobs_problem, rast, re2, job)
        end

        # Again there are no jobs left
        re3 = ConScape.reassess(nested_jobs_problem, assessment)
        @test re3.njobs == 0
        @test length(re3.indices) == 0
        @test count(re3.mask) == 0
    end

    @test keys(windowed_result) == 
          keys(nested_result) == 
          keys(batch_result) == 
          keys(batch_jobs_result) == 
          keys(nested_jobs_result) == 
          Tuple(sort(collect(expected_layers)))

    # These may be approximate after mosaic order changes
    compare(a, b) = isnan(a) && isnan(b) || isapprox(a, b)
    
    @test all(batch_jobs_result.ch .=== batch_result.ch)
    @test all(batch_jobs_result.betk .=== batch_result.betk)
    @test all(compare.(permutedims(batch_result.ch), windowed_result.ch))
    @test all(compare.(permutedims(batch_result.betk), windowed_result.betk))
    @test all(compare.(nested_result.betk, nested_jobs_result.betk))
    @test all(compare.(nested_result.ch, nested_jobs_result.ch))

    # TODO: there are some tiny fp differences in the nested result
    @test all(map(nested_result.ch, batch_result.ch) do n, b
        isnan(n) && isnan(b) || isapprox(n, b)
    end)

    # plot(windowed_result)
    # plot(batch_result)
    # plot(batch_jobs_result)
    # plot(nested_result)
    # plot(nested_jobs_result)
end
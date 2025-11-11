using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

θ = 0.1
landscape = "sno_2000"
# The way the ascii is read in is reversed and rotated from what GDAL does
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

measures = (;
    betm=ConScape.MovementFlow(),
    # betq=Betweenness(QualityWeighted()), # Doesn't work windowed!
    ch=FunctionalHabitat(),
    # # TODO sens=ConScape.Sensitivity(),
)
# Set low alpha here so the decay is steep for testing
distance_transformation = x -> exp(-x / 2)
movement = RandomisedShortestPath(ExpectedCost(); 
    theta=θ, distance_transformation
)

solver = ConScape.VectorSolver()
problem = ConScapeProblem(; measures, movement, solver);
solve(problem, rast; verbose=true)

@testset "window shape" begin
    circle_windowed_problem = WindowedProblem(problem; 
        buffer=10, centersize=5, threaded=true, test_windows=true, shape=:circle
    )
    square_windowed_problem = WindowedProblem(problem; 
        buffer=10, centersize=5, threaded=true, test_windows=true, shape=:square
    )
    wi = init(circle_windowed_problem, rast)
    circle_st = ConScape._get_window_with_zeroed_buffer(circle_windowed_problem, rast, wi.ranges[6])
    square_st = ConScape._get_window_with_zeroed_buffer(square_windowed_problem, rast, wi.ranges[6])
    # Affinities never have rounded corners
    @test circle_st.steplikelihood[end] !== 0.0
    @test square_st.steplikelihood[end] !== 0.0
    # Source qualities do for :circle 
    @test circle_st.sourcequality[end] === 0.0
    # But not for :square
    @test square_st.sourcequality[end] !== 0.0
end

@testset "target mosaicing matches original" begin
    windowed_problem = WindowedProblem(problem; 
        buffer=10, centersize=5, threaded=true, test_windows=true
    )
    @test collect(ConScape.window_ranges(windowed_problem, rast)) == [
        (1:25, 1:25)   (1:25, 6:30)   (1:25, 11:35)   (1:25, 16:40)   (1:25, 21:45)   (1:25, 26:50)   (1:25, 31:55)   (1:25, 36:59)
        (6:30, 1:25)   (6:30, 6:30)   (6:30, 11:35)   (6:30, 16:40)   (6:30, 21:45)   (6:30, 26:50)   (6:30, 31:55)   (6:30, 36:59)
        (11:35, 1:25)  (11:35, 6:30)  (11:35, 11:35)  (11:35, 16:40)  (11:35, 21:45)  (11:35, 26:50)  (11:35, 31:55)  (11:35, 36:59)
        (16:40, 1:25)  (16:40, 6:30)  (16:40, 11:35)  (16:40, 16:40)  (16:40, 21:45)  (16:40, 26:50)  (16:40, 31:55)  (16:40, 36:59) 
        (21:44, 1:25)  (21:44, 6:30)  (21:44, 11:35)  (21:44, 16:40)  (21:44, 21:45)  (21:44, 26:50)  (21:44, 31:55)  (21:44, 36:59) 
    ]
    test_results = solve(windowed_problem, rast)
    inner_targets = copy(rast.quality)
    replace!(inner_targets, NaN => 0.0)
    # Edge targets are lost with windowing
    inner_targets[1:10, :] .= 0
    inner_targets[:, 1:10] .= 0
    inner_targets[end-9:end, :] .= 0
    inner_targets[:, end-9:end] .= 0
    keys(test_results)
    @test parent(inner_targets) == parent(test_results.targetquality)
end

@testset "windowed results approximate non-windowed" begin
    buffer=15
    windowed_problem = WindowedProblem(problem; 
        buffer, centersize=2, shape=:circle
    )
    mask!(rast; with=rast)
    wi = init(windowed_problem, rast)
    rast_inner = ConScape._get_window_with_zeroed_buffer(wi; shape=:square)
    @time wp_result = solve!(wi)
    @time p_result = solve(problem, rast_inner)
    @test maplayers(p_result, wp_result) do P, WP
        broadcast(P, WP) do p, wp
            isnan(p) && isnan(wp) || isapprox(p, wp; atol=1e-4)
        end |> all
    end |> all
end


# BatchProblem writes files to disk and mosaics to RasterStack
@testset "batch problem matches windowed problem" begin
    solver = VectorSolver()
    # Use a higher alpha to catch differences
    distance_transformation = x -> exp(-x / 50)
    movement = RandomisedShortestPath(ExpectedCost(); theta=θ, distance_transformation)
    problem = ConScapeProblem(; measures, movement, solver);

    kw = (; buffer=10, centersize=5)
    windowed_problem = WindowedProblem(problem; kw...)
    @time windowed_init = init(windowed_problem, rast);
    @test windowed_init isa ConScape.WindowedInit
    @time windowed_result = solve!(windowed_init);

    batch_problem = BatchProblem(problem; datapath=tempname(), kw...)
    paths = solve(batch_problem, rast; verbose=true)
    Rasters.mosaic(sum, RasterStack.(paths))

    batch_result = mosaic(batch_problem; to=rast)
    @test batch_result isa RasterStack

    batch_init_problem = BatchProblem(problem; datapath=tempname(), kw...)
    batch_init = init(batch_init_problem, rast; verbose=true)
    paths = solve!(batch_init)

    batch_init_result = mosaic(sum, RasterStack.(paths))
    @test batch_result isa RasterStack


    # BatchProblem can be run as batch jobs for clusters
    # We just need a new path to make sure the result is from a new run
    batch_jobs_problem = BatchProblem(problem; 
        datapath=tempname(), kw...
    )
    assessment = ConScape.assess(batch_jobs_problem, rast)
    @test assessment.njobs == 39

    paths = solve(batch_jobs_problem, rast, assessment, 1; verbose=true)
    @test keys(paths) == (:betm, :ch)
    for job in 1:assessment.njobs
        solve(batch_jobs_problem, rast, assessment, job)
    end
    batch_jobs_result = mosaic(batch_jobs_problem; to=rast)

    batch_jobs_init_problem = BatchProblem(problem; datapath=tempname(), kw...)
    assessment = ConScape.assess(batch_jobs_init_problem, rast)
    for job in 1:assessment.njobs
        batch_jobs_init = init(batch_jobs_init_problem, rast, assessment; verbose=true)
        solve(batch_jobs_init, job; verbose=true)
    end
    batch_jobs_init_result = mosaic(sum, batch_jobs_init_problem, rast)

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

    nested_problem = BatchProblem(windowed_problem; 
        datapath=tempname(), centersize=(10, 10)
    )
    paths = solve(nested_problem, rast)
    @test keys(paths[1]) == (:betm, :ch)
    @test paths[1].betm isa String
    nested_result = mosaic(sum, nested_problem; to=rast)
    @test nested_result isa RasterStack

    nested_jobs_problem = ConScape.BatchProblem(windowed_problem; 
        datapath=tempname(), centersize=(10, 10)
    )
    assessment = ConScape.assess(nested_jobs_problem, rast);
    # Try one
    @time nested_batch_init = init(nested_jobs_problem, rast, assessment)
    @time solve(nested_batch_init, 5; verbose=true)
    res = solve(windowed_problem, rast; mosaic_return=false)
    for job in 1:assessment.njobs
        solve(nested_jobs_problem, rast, job)
    end
    nested_jobs_result = mosaic(sum, nested_jobs_problem; to=rast)

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
          keys(batch_jobs_init_result) == 
          keys(nested_jobs_result) == 
          keys(measures)

    # These may be approximate after mosaic order changes
    compare(a, b) = ismissing(a) && ismissing(b) || isnan(a) && isnan(b) || isapprox(a, b)
    sts = RasterStack.(filter(isdir, ConScape.batch_paths(batch_jobs_problem, rast)))

    @test all(batch_jobs_result.ch .=== batch_result.ch)
    @test all(batch_jobs_result.betm .=== batch_result.betm)
    @test all(batch_jobs_init_result.ch .=== batch_result.ch)
    @test all(batch_jobs_init_result.betm .=== batch_result.betm)
    @test all(compare.(nested_result.betm, nested_jobs_result.betm))
    @test all(compare.(nested_result.ch, nested_jobs_result.ch))
    @test all(compare.(permutedims(nested_result.betm), windowed_result.betm))
    @test all(compare.(permutedims(nested_result.ch), windowed_result.ch))
    @test all(compare.(permutedims(nested_jobs_result.betm), windowed_result.betm))
    @test all(compare.(permutedims(nested_jobs_result.ch), windowed_result.ch))
    @test all(compare.(permutedims(batch_result.ch), windowed_result.ch))
    @test all(compare.(permutedims(batch_result.betm), windowed_result.betm))

    # plot(windowed_result)
    # plot(batch_result)
    # plot(batch_jobs_result)
    # plot(nested_result)
    # plot(nested_jobs_result)
end

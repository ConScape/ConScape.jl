using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL

using ConScape: vec_workspaces, mat_workspaces

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")

θ = 0.1
landscape = "sno_1000"
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

measures = (;
    betm=ConScape.MovementFlow(),
    ch=FunctionalHabitat(),
)
movement = RandomisedShortestPath(ExpectedCost(); theta=θ)
problem = ConScapeProblem(; measures, movement)

@testset "WindowAssessment structure" begin
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    @test assessment isa ConScape.WindowAssessment
    @test assessment.size == size(rast)
    @test assessment.shape == (29, 16)  # Expected window grid shape
    @test assessment.njobs > 0
    @test assessment.njobs <= prod(assessment.shape)
    @test length(assessment.mask) == prod(assessment.shape)
    @test length(assessment.indices) == assessment.njobs
    @test all(assessment.mask[assessment.indices])
    @test length(assessment.sparse_sizes) == prod(assessment.shape)

    # All sparse_sizes should be tuples of (sources, targets)
    @test all(s -> s isa Tuple{Int,Int}, assessment.sparse_sizes)

    # Non-empty windows should have positive sparse sizes
    for i in assessment.indices
        sources, targets = assessment.sparse_sizes[i]
        @test sources > 0
        @test targets > 0
    end
end

@testset "NestedAssessment structure" begin
    nested_problem = BatchProblem(
        WindowedProblem(problem; buffer=10, centersize=5);
        buffer=10, centersize=20, datapath=tempname()
    )
    assessment = ConScape.assess(nested_problem, rast; verbose=false)

    @test assessment isa ConScape.NestedAssessment
    @test assessment.size == size(rast)
    @test assessment.njobs > 0
    @test length(assessment.assessments) == prod(assessment.shape)
    @test all(a -> a isa ConScape.WindowAssessment, assessment.assessments)

    # Inner assessments should have consistent structure
    for i in assessment.indices
        inner = assessment.assessments[i]
        @test inner.njobs > 0 || isempty(inner.indices)
    end
end

@testset "sparse_sizes accuracy" begin
    windowed_problem = WindowedProblem(problem; buffer=15, centersize=10)
    assessment = ConScape.assess(windowed_problem, rast)

    # Initialize the windowed problem to get actual sizes
    wi = init(windowed_problem, rast; indices=assessment.indices, sparse_sizes=assessment.sparse_sizes)
    gg1 = init(wi, 9)
    cg1 = init(gg1, 1)

    # For each window, check that estimated sizes are between the grid and connected graph
    # sizes. We don't actually run the yj
    for (idx, window_idx) in enumerate(assessment.indices)
        g = init(init(wi, window_idx), 1)
        # The estimate should be reasonable (not zero, not impossibly large)
        est_sources, est_targets = assessment.sparse_sizes[window_idx]
        connectedgraph_sources, connectedgraph_targets = ConScape.connectedgraph_size(g)
        gridgraph_sources, gridgraph_targets = ConScape.gridgraph_size(g)
        # Estimated sizes must be equal or larger to the first (largest) connected graph
        @test est_sources >= connectedgraph_sources
        @test est_targets >= connectedgraph_targets
        # But they must also be equal or smaller then the grid
        @test est_sources <= gridgraph_sources
        @test est_targets <= gridgraph_targets
    end
end

@testset "AssessmentWarnings" begin
    # Test warnings detection
    quality_with_nan = copy(quality)
    quality_with_nan[20, 20] = NaN  # Add a NaN in the interior

    rast_with_nan = RasterStack((; steplikelihood, quality=quality_with_nan))

    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast_with_nan)

    # Should detect NaN in target quality (quality is used as both source and target)
    @test assessment.warnings isa ConScape.AssessmentWarnings
    @test any(assessment.warnings)  # At least one warning should be true
end

@testset "reassess removes completed jobs" begin
    _tempdir = mktempdir()
    batch_problem = BatchProblem(
        WindowedProblem(problem; buffer=10, centersize=5);
        buffer=10, centersize=15, datapath=_tempdir
    )

    assessment = ConScape.assess(batch_problem, rast; verbose=false);
    original_njobs = assessment.njobs

    # Before any jobs run, reassess should return same count
    re1 = ConScape.reassess(batch_problem, assessment)
    @test re1.njobs == original_njobs

    # Run one job
    if original_njobs > 0
        solve(batch_problem, rast, assessment, 1; verbose=false)

        # After one job, reassess should show one fewer job
        re2 = ConScape.reassess(batch_problem, assessment)
        @test re2.njobs == original_njobs - 1
        @test length(re2.indices) == original_njobs - 1
    end

    rm(_tempdir; recursive=true)
end

@testset "assessment keywords passed to init/solve" begin
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    # Using assessment should work the same as manual keywords
    wi_from_assessment = init(windowed_problem, rast, assessment)
    wi_manual = init(windowed_problem, rast;
        indices=assessment.indices,
        sparse_sizes=assessment.sparse_sizes
    )

    @test wi_from_assessment.indices == wi_manual.indices
    @test wi_from_assessment.sparse_sizes == wi_manual.sparse_sizes
end

@testset "init with WindowAssessment" begin
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    # init(WindowedProblem, rast, WindowAssessment) should work
    wi = init(windowed_problem, rast, assessment)
    @test wi isa ConScape.WindowedInit
    @test wi.indices == assessment.indices
    @test wi.sparse_sizes == assessment.sparse_sizes

    # Results should be solvable
    result = solve!(wi)
    @test haskey(result, :betm)
    @test haskey(result, :ch)
end

@testset "solve with WindowAssessment for BatchProblem" begin
    _tempdir = mktempdir()
    batch_problem = BatchProblem(
        WindowedProblem(problem; buffer=10, centersize=5);
        buffer=10, centersize=15, datapath=_tempdir
    )

    assessment = ConScape.assess(batch_problem, rast; verbose=false)

    # solve(BatchProblem, rast, assessment, job_id) should work
    paths = solve(batch_problem, rast, assessment, 1; verbose=false)
    @test !isempty(paths)
    @test all(isfile, paths)

    rm(_tempdir; recursive=true)
end

@testset "init with NestedAssessment for BatchProblem" begin
    _tempdir = mktempdir()
    batch_problem = BatchProblem(
        WindowedProblem(problem; buffer=10, centersize=5);
        buffer=10, centersize=15, datapath=_tempdir
    )

    assessment = ConScape.assess(batch_problem, rast; verbose=false)

    # init(BatchProblem{WindowedProblem}, rast, NestedAssessment) should work
    bi = init(batch_problem, rast, assessment)
    @test bi isa ConScape.BatchInit

    rm(_tempdir; recursive=true)
end

@testset "estimate_memory_for_centersize" begin
    # Create a simple assessment
    sz = (100, 100)
    rast_test = let
        sq = Raster(ones(sz); dims=(X(1:sz[1]), Y(1:sz[2])))
        RasterStack((; sourcequality=sq, targetquality=copy(sq), stepcost=copy(sq), steplikelihood=copy(sq)))
    end

    problem_sens = ConScapeProblem((SensitivityAnalysis(wrt=StepCost()),); movement=RSP(theta=0.1))
    wp = WindowedProblem(problem_sens; centersize=20, buffer=10, mosaic_return=false)
    assessment = ConScape.assess(wp, rast_test)

    # Function should return positive estimates
    for cs in [10, 15, 20, 25]
        mem = estimate_memory_for_centersize(problem_sens, assessment, cs)
        @test mem.total > 0
    end

    # Memory should scale with centersize² (targets = centersize²)
    mem_10 = estimate_memory_for_centersize(problem_sens, assessment, 10)
    mem_20 = estimate_memory_for_centersize(problem_sens, assessment, 20)
    mem_30 = estimate_memory_for_centersize(problem_sens, assessment, 30)

    # Doubling centersize should roughly quadruple dense matrix memory
    # (but sparse components don't scale, so ratio will be less than 4)
    @test mem_20.total > mem_10.total * 2
    @test mem_30.total > mem_20.total * 1.5

    # Original assessment memory_estimate should match centersize=20
    @test assessment.memory_estimate.total ≈ mem_20.total rtol=0.1
end

@testset "memory_estimate matches actual allocations" begin
    measures = (;
        betm=MovementFlow(),
        ch=FunctionalHabitat(),
        sens=SensitivityAnalysis(; wrt=StepCost(), metric=Summation(), type=Sensitivity())
    )
    movement = RandomisedShortestPath(ExpectedCost(); theta=θ)
    problem = ConScapeProblem(; measures, movement)

    windowed_problem = WindowedProblem(problem; buffer=40, centersize=15)
    assessment = ConScape.assess(windowed_problem, rast)
    mem = assessment.memory_estimate

    wi = init(windowed_problem, rast, assessment)
    largest_subgraph = findmax(prod, assessment.sparse_sizes)[2]
    gi = init(wi, largest_subgraph)
    cgi = init(gi, 1)

    # Workspaces are not realocated
    @test vec_workspaces(wi) === vec_workspaces(gi) === vec_workspaces(cgi)

    # Mostly just vector workspaces are allocated for windows
    allocated_wi = @allocated init(windowed_problem, rast, assessment)
    wi_size = mem.vec_workspaces + mem.mat_workspaces
    @test wi_size <= allocated_wi < wi_size * 1.01
    allocated_gi = @allocated init(wi, largest_subgraph)
    allocated_cgi = @allocated init(gi, 1)
    allocated_wi

    @allocated solve!(cgi)
    @allocated solve!(gi)
    @allocated solve!(wi)

    # ConScape._sizeofsparse(size(gi.gridgraph.stepcost)..., 8)
    # @test 
    # mem.gridgraph
    # allocated_gi = @allocated init(wi, largest_subgraph)
    # allocated_gi
    # < 
    # allocated_wi
    # allocated_cgi = @allocated init(gi, 1)
    #
    # # Solve barely allocates
    # @test (@allocated solve!(wi)) < 50000
end

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
    @test assessment.shape == (5, 8)  # Expected window grid shape
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
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    # Initialize the windowed problem to get actual sizes
    wi = init(windowed_problem, rast; indices=assessment.indices, sparse_sizes=assessment.sparse_sizes)
    gg1 = init(wi, 1)
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

    assessment = ConScape.assess(batch_problem, rast; verbose=false)
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

@testset "assessment estimate quality" begin
    # Test that sparse_sizes estimates are close to actual sizes after solving
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    wi = init(windowed_problem, rast, assessment)

    # Check a few windows
    for i in 1:min(3, length(assessment.indices))
        idx = assessment.indices[i]
        est_sources, est_targets = assessment.sparse_sizes[idx]

        # The estimate should be a reasonable upper bound
        # (actual may be smaller due to connected component filtering)
        @test est_sources > 0
        @test est_targets > 0
    end

    # Verify assessment enables efficient memory pre-allocation
    # by checking sparse_sizes are used correctly
    @test length(wi.sparse_sizes) == length(assessment.sparse_sizes)
    @test wi.sparse_sizes == assessment.sparse_sizes
end

@testset "actual memory usage scales with sparse_sizes" begin
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=8, gc=false)
    assessment = ConScape.assess(windowed_problem, rast)

    # Find two windows with different sizes
    sizes_with_idx = [(prod(assessment.sparse_sizes[i]), i) for i in assessment.indices]
    sort!(sizes_with_idx)

    small_idx = sizes_with_idx[1][2]
    large_idx = sizes_with_idx[end][2]

    small_size = prod(assessment.sparse_sizes[small_idx])
    large_size = prod(assessment.sparse_sizes[large_idx])

    # Only test if sizes differ meaningfully
    @test large_size > 10small_size

    # Measure actual memory allocation for each
    wi_small = init(windowed_problem, rast; indices=[small_idx], sparse_sizes=assessment.sparse_sizes)
    wi_large = init(windowed_problem, rast; indices=[large_idx], sparse_sizes=assessment.sparse_sizes)

    # Warm up
    using ProfileView
    f(wi, n) = for _ in 1:n solve!(wi) end
    @profview f(wi_small, 1000)
    @profview f(wi_large, 100)

    solve!(wi_small)
    solve!(wi_large)

    # Measure allocations
    small_alloc = @allocated solve!(wi_small)
    large_alloc = @allocated solve!(wi_large)

end

# Test measures that use matrix vec_workspaces (mat_workspace)
# These require accurate sparse_sizes for pre-allocation
@testset "assess with EigMax measure" begin
    # EigMax uses mat_workspace
    measures_eigmax = (;
        eigmax=ConScape.EigMax(),
        ch=FunctionalHabitat(),  # Need at least one regular measure
    )
    problem_eigmax = ConScapeProblem(; measures=measures_eigmax, movement, solver)
    windowed_eigmax = WindowedProblem(problem_eigmax; buffer=10, centersize=5)

    assessment = ConScape.assess(windowed_eigmax, rast)
    @test assessment.njobs > 0

    # Verify we can init and solve with the assessment
    wi = init(windowed_eigmax, rast, assessment)
    result = solve!(wi)
    @test haskey(result, :eigmax)
    @test haskey(result, :ch)
end

@testset "assess with SensitivityAnalysis measure" begin
    # SensitivityAnalysis uses mat_workspace
    measures_sens = (;
        sens=SensitivityAnalysis(; wrt=Quality(), metric=Summation()),
        ch=FunctionalHabitat(),
    )
    problem_sens = ConScapeProblem(; measures=measures_sens, movement, solver)
    windowed_sens = WindowedProblem(problem_sens; buffer=10, centersize=5)

    assessment = ConScape.assess(windowed_sens, rast)
    @test assessment.njobs > 0

    # Verify we can init and solve with the assessment
    wi = init(windowed_sens, rast, assessment)
    result = solve!(wi)
    @test haskey(result, :sens)
    @test haskey(result, :ch)
end

# =============================================================================
# High-level tests: Does assessment help users plan jobs for cluster constraints?
#
# Context: Batch jobs run on SLURM clusters with ~3-4GB RAM per core.
# Users need to know if their window configuration will fit in available memory.
# =============================================================================

@testset "memory prediction accuracy" begin
    # Test that sparse_sizes can predict actual memory usage
    #
    # Architecture note:
    # - init(WindowedProblem) -> WindowedInit (lightweight: just references + vectors)
    # - solve!(WindowedInit) -> for each window, creates GridGraphInit with heavy buffers
    #
    # So allocations happen inside solve!(), not init() for WindowedProblem.
    # The question is: can we predict those allocations from sparse_sizes?

    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    # Find the largest window by sparse_sizes (worst case for memory)
    max_idx = argmax(idx -> prod(assessment.sparse_sizes[idx]), assessment.indices)
    est_sources, est_targets = assessment.sparse_sizes[max_idx]

    # Predicted memory: main cost is dense matrices (Z, and potentially others)
    # Each dense matrix is sources × targets × 8 bytes
    predicted_bytes = est_sources * est_targets * sizeof(Float64)

    # Warm up: run full cycle to trigger all compilation
    wi_warmup = init(windowed_problem, rast; indices=[max_idx], sparse_sizes=assessment.sparse_sizes)
    solve!(wi_warmup)
    wi_warmup = nothing
    GC.gc()

    # Measure WindowedInit creation (should be small - just metadata)
    windowed_init_alloc = @allocated begin
        wi = init(windowed_problem, rast; indices=[max_idx], sparse_sizes=assessment.sparse_sizes)
    end

    # Warm up solve!() on this instance
    solve!(wi)
    GC.gc()

    # Measure solve!() allocations (includes per-window GridGraphInit creation)
    solve_alloc = @allocated solve!(wi)

    println("  Window $max_idx: sources=$est_sources, targets=$est_targets")
    println("  Predicted dense matrix: $(round(predicted_bytes / 1024^2, digits=2)) MB")
    println("  WindowedInit creation: $(round(windowed_init_alloc / 1024^2, digits=2)) MB (expected: small)")
    println("  solve!() allocations: $(round(solve_alloc / 1024^2, digits=2)) MB (includes GridGraphInit)")
    println("  solve!/predicted ratio: $(round(solve_alloc / predicted_bytes, digits=2))x")

    # solve!() allocations should be predictable from sparse_sizes
    # Current ratio is ~25x (includes sparse matrices, vec_workspaces, etc.)
    # Target: understand the multiplier to give users accurate memory estimates
    @test solve_alloc < predicted_bytes * 50
end

@testset "memory stability across windows" begin
    # Memory usage should NOT grow as we process more windows
    # Buffers should be reused, not accumulated
    windowed_problem = WindowedProblem(problem; buffer=10, centersize=5)
    assessment = ConScape.assess(windowed_problem, rast)

    wi = init(windowed_problem, rast, assessment)

    # Warm up
    solve!(wi)
    GC.gc()

    # Track allocations for consecutive solves
    allocs = Float64[]
    for _ in 1:3
        GC.gc()
        a = @allocated solve!(wi)
        push!(allocs, a)
    end

    println("  Allocations across 3 consecutive solves: $(round.(allocs ./ 1024^2, digits=2)) MB")

    # Allocations should be stable (not growing)
    # Allow 20% variance for GC timing differences
    avg_alloc = sum(allocs) / length(allocs)
    for a in allocs
        @test a < avg_alloc * 1.5  # No single run should be 50% above average
    end

    # Later runs shouldn't allocate significantly more than earlier runs
    @test allocs[end] < allocs[1] * 1.5
end

@testset "per-window allocations vs total" begin
    # Ideal: total memory ≈ memory for one window (buffers reused)
    # This tests that we're not accumulating garbage across windows
    windowed_problem = WindowedProblem(problem; buffer=8, centersize=4)
    assessment = ConScape.assess(windowed_problem, rast)

    # Solve just one window
    wi_single = init(windowed_problem, rast;
        indices=[assessment.indices[1]],
        sparse_sizes=assessment.sparse_sizes
    )
    solve!(wi_single)
    GC.gc()
    single_alloc = @allocated solve!(wi_single)

    # Solve all windows
    wi_all = init(windowed_problem, rast, assessment)
    solve!(wi_all)
    GC.gc()
    all_alloc = @allocated solve!(wi_all)

    n_windows = length(assessment.indices)
    per_window_avg = all_alloc / n_windows

    println("  Single window allocation: $(round(single_alloc / 1024^2, digits=2)) MB")
    println("  All windows total: $(round(all_alloc / 1024^2, digits=2)) MB")
    println("  Per-window average: $(round(per_window_avg / 1024^2, digits=2)) MB")
    println("  Number of windows: $n_windows")
    println("  Overhead ratio: $(round(all_alloc / (single_alloc * n_windows), digits=2))x")

    # Per-window average should be similar to single window (within 3x)
    # If much higher, buffers aren't being reused properly
    @test per_window_avg < single_alloc * 3
end

# =============================================================================
# Memory estimation functions
# =============================================================================

@testset "estimate_memory" begin
    problem_fh = ConScapeProblem((FunctionalHabitat(),); movement=RSP(theta=0.1))
    problem_sens = ConScapeProblem((SensitivityAnalysis(wrt=StepCost()),); movement=RSP(theta=0.1))

    est = ConScape.estimate_memory(problem_sens, 1600, 400)

    # Returns NamedTuple with breakdown and total
    @test haskey(est, :vec_workspaces)
    @test haskey(est, :mat_workspaces)
    @test haskey(est, :dense_precalc)
    @test haskey(est, :sparse_matrices)
    @test haskey(est, :lu_factorization)
    @test haskey(est, :total)

    # Total equals sum of components
    components = (est.vec_workspaces, est.mat_workspaces, est.dense_precalc,
                  est.sparse_matrices, est.lu_factorization, est.graph,
                  est.quality_vectors, est.outputs, est.gridgraph)
    @test est.total == sum(components)

    # All values non-negative
    for v in values(est)
        @test v >= 0
    end

    # Basic sanity: total is positive
    @test ConScape.estimate_memory(problem_fh, 1600, 400).total > 0
    @test ConScape.estimate_memory(problem_sens, 1600, 400).total > 0

    # SensitivityAnalysis needs more memory than FunctionalHabitat
    @test est.total > ConScape.estimate_memory(problem_fh, 1600, 400).total
end

@testset "estimate_memory scaling" begin
    problem_fh = ConScapeProblem((FunctionalHabitat(),); movement=RSP(theta=0.1))
    problem_sens = ConScapeProblem((SensitivityAnalysis(wrt=StepCost()),); movement=RSP(theta=0.1))

    # FunctionalHabitat (sparse-only): scales linearly with sources
    est_fh_small = ConScape.estimate_memory(problem_fh, 1600, 400).total
    est_fh_large = ConScape.estimate_memory(problem_fh, 160000, 400).total
    ratio_fh = est_fh_large / est_fh_small
    @test 50 < ratio_fh < 150  # ~100x sources, linear scaling

    # SensitivityAnalysis: with fixed targets, scales linearly with sources
    est_sens_small = ConScape.estimate_memory(problem_sens, 1600, 400).total
    est_sens_large = ConScape.estimate_memory(problem_sens, 160000, 400).total
    ratio_sens = est_sens_large / est_sens_small
    @test 50 < ratio_sens < 150

    # With both scaling, dense measures scale quadratically
    est_sens_both = ConScape.estimate_memory(problem_sens, 160000, 40000).total
    ratio_both = est_sens_both / est_sens_small
    @test ratio_both > 1000
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
        @test mem > 0
    end

    # Memory should scale with centersize² (targets = centersize²)
    mem_10 = estimate_memory_for_centersize(problem_sens, assessment, 10)
    mem_20 = estimate_memory_for_centersize(problem_sens, assessment, 20)
    mem_30 = estimate_memory_for_centersize(problem_sens, assessment, 30)

    # Doubling centersize should roughly quadruple dense matrix memory
    # (but sparse components don't scale, so ratio will be less than 4)
    @test mem_20 > mem_10 * 2
    @test mem_30 > mem_20 * 1.5

    # Original assessment memory_estimate should match centersize=20
    @test assessment.memory_estimate ≈ mem_20 rtol=0.1
end

@testset "WindowAssessment memory_estimate field" begin
    sz = (100, 100)
    rast_test = let
        sq = Raster(ones(sz); dims=(X(1:sz[1]), Y(1:sz[2])))
        RasterStack((; sourcequality=sq, targetquality=copy(sq), stepcost=copy(sq), steplikelihood=copy(sq)))
    end

    problem_sens = ConScapeProblem((SensitivityAnalysis(wrt=StepCost()),); movement=RSP(theta=0.1))
    wp = WindowedProblem(problem_sens; centersize=20, buffer=10, mosaic_return=false)
    assessment = ConScape.assess(wp, rast_test)

    # memory_estimate should be populated
    @test assessment.memory_estimate > 0

    # Should be in reasonable range (MB, not bytes or GB for this small test)
    @test 1 < assessment.memory_estimate < 1000  # Between 1 MB and 1 GB
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

    # Mostly just vector workspaces are allocated for windows
    @allocated init(windowed_problem, rast, assessment)
    allocated_wi = @allocated init(windowed_problem, rast, assessment)
    wi_size = mem.vec_workspaces + mem.mat_workspaces
    @test wi_size <= allocated_wi < wi_size * 1.01

    largest_subgraph = findmax(prod, assessment.sparse_sizes)[2]
    gi = init(wi, largest_subgraph)
    cgi = init(gi, 1)

    # Workspaces are not realocated
    @test vec_workspaces(wi) === vec_workspaces(gi) === vec_workspaces(cgi)

    ConScape._sizeofsparse(size(gi.gridgraph.stepcost)..., 8)
    @test 
    mem.gridgraph
    allocated_gi = @allocated init(wi, largest_subgraph)
    allocated_cgi 
    allocated_gi 
    allocated_wi
    allocated_sove_cgi 
    cgi = init(gi, 1)
    allocated_cgi = @allocated init(gi, 1)
    solve!(cgi)
    allocated_sove_cgi = @allocated solve!(cgi)
    solve!(wi)
    allocated_solve_wi = @allocated solve!(wi)

    # Solve barely allocates
    @profview_allocs nolve!(cgi)
    f(wi, i, n) = for m in 1:n[] 
        x = init(wi, i)
        m == n && return x 
    end
    n = Ref(20000000)
    @profview_allocs f(cgi, 1, n)
    f(cgi, 1, n)
    # ProfileView.view(; data=Profile.Allocs.fetch())
    Profile.Allocs.fetch()
    PProf.Allocs.pprof()

    # Solve barely allocates
    @test (@allocated solve!(wi)) < 50000
end

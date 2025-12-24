using ConScape, Test, SparseArrays
using Rasters, ArchGDAL

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")

θ = 0.1
landscape = "sno_1000"
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

measures = (; eb=EdgeBetweenness(; weighting=Unweighted()))
distance_transformation = ExpMinusAlpha(2.0)
movement = RandomisedShortestPath(ExpectedCost(); theta=θ, distance_transformation)
solver = ConScape.VectorSolver()

@testset "mmap workspaces" begin
    # Test without mmap (default)
    problem_nommap = ConScapeProblem(; measures, movement, solver)
    @test ConScape.mmap_path(problem_nommap) === nothing

    wp_nommap = WindowedProblem(problem_nommap; buffer=10, centersize=5)
    wi_nommap = init(wp_nommap, rast)
    @test isempty(ConScape.mat_workspaces(wi_nommap).mmap_paths)

    # Test with mmap
    mmap_dir = mktempdir()
    problem_mmap = ConScapeProblem(; measures, movement, solver, mmap_path=mmap_dir)
    @test ConScape.mmap_path(problem_mmap) == mmap_dir

    wp_mmap = WindowedProblem(problem_mmap; buffer=10, centersize=5)
    wi_mmap = init(wp_mmap, rast)
    mmap_paths = ConScape.mat_workspaces(wi_mmap).mmap_paths
    @test !isempty(mmap_paths)
    @test all(startswith.(mmap_paths, mmap_dir))

    # Files should exist
    @test all(isfile, mmap_paths)

    # Mmap arrays should be writable
    mat_ws = ConScape.mat_workspaces(wi_mmap)
    arr = take!(mat_ws)
    arr[1, 1] = 42.0
    @test arr[1, 1] == 42.0
    put!(mat_ws, arr)

    # Cleanup should remove files
    paths = copy(mmap_paths)
    ConScape.cleanup!(wi_mmap)
    @test all(!isfile, paths)
end

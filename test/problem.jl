using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL, NCDatasets, Plots
using LinearSolve

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
# They are only the same for Float32
@test all(Float32.(affinities_asc) .=== Float32.(rast.affinities))
@test all(Float32.(qualities_asc) .=== Float32.(rast.qualities))

graph_measures = graph_measures = (;
    ch=ConScape.ConnectedHabitat(),
    betq=ConScape.BetweennessQweighted(),
    betk=ConScape.BetweennessKweighted(),
    # # TODO sens=ConScape.Sensitivity(),
    ebetq=ConScape.EdgeBetweennessQweighted(),
    ebetk=ConScape.EdgeBetweennessKweighted(),
    mkld=ConScape.MeanKullbackLeiblerDivergence(),
    mlcd=ConScape.MeanLeastCostKullbackLeiblerDivergence(),
    eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
distance_transformation = (nodist=nothing, one=one, exp50=t -> exp(-t/50))
connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)

expected_layers = (
    :ch_nodist, :ch_one, :ch_exp50, 
    :betq, 
    :betk_nodist, :betk_one, :betk_exp50, 
    :ebetq, 
    :ebetk_nodist, :ebetk_one, :ebetk_exp50, 
    :mkld, 
    :mlcd,
    :eigmax_nodist, :eigmax_one, :eigmax_exp50, 
)
affinities_sparse = ConScape.graph_matrix_from_raster(affinities)
test_g = ConScape.Grid(size(affinities)...;
    affinities=affinities_sparse,
    qualities
)
test_grsp = ConScape.GridRSP(test_g; θ)

solvers = (
    ConScape.MatrixSolver(),
    ConScape.VectorSolver(),
    ConScape.VectorSolver(; threaded=true),
    ConScape.LinearSolver(),
)
solver = ConScape.VectorSolver(; threaded=true)
solver = ConScape.MatrixSolver()

for solver in solvers
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem = ConScape.Problem(; 
        graph_measures, connectivity_measure, solver,
    )
    @time workspace = init(problem, rast);
    @testset "initialised grids are the same" begin
        @test workspace.grsp.W == test_grsp.W
        @test workspace.grsp.Z == test_grsp.Z
        @test workspace.grsp.Pref == test_grsp.Pref
        @test workspace.grsp.θ == test_grsp.θ
        foreach(propertynames(test_g)) do n
            @test isequal(getproperty(workspace.grid, n), getproperty(test_g, n))
        end
        @test workspace.expected_costs == ConScape.expected_cost(test_grsp)
        @test workspace.free_energy_distances == ConScape.free_energy_distance(test_grsp)
    end

    ConScape.allocations(problem, rast).total / 1e6

    @time result = ConScape.solve(problem, workspace);
    # @profview result = ConScape.solve(problem, workspace)
    @test result isa NamedTuple
    @test size(result.ch_one) == size(rast)
    @test keys(result) == expected_layers
    g = workspace.grid
    # Base.summarysize(workspace) / 1e6
    # ConScape.allocations(problem, size(workspace.B_sparse)).total / 1e6

    @testset "Test mean_kl_divergence" begin
        @test ConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
        @test result.mkld[] ≈ 323895.3828183995
    end

    @testset "mean_lc_kl_divergence" begin
        @test result.mlcd[] ≈ 1.5660600315073947e6
    end
    @testset "q-weighted" begin
        @test result.betq isa Raster
        @test isapprox(result.betq[21:23, 21:23], [
            1930.1334372152335  256.91061166392745 2866.2998374065373
            4911.996715311025  1835.991238248377    720.755518530375
            4641.815380725279  3365.3296878569213   477.1085971945757], atol=1e-3)
    end
    @testset "k-weighted" begin
        @test result.betk_nodist isa Raster
        bet = ConScape.betweenness_kweighted(test_grsp)
        @test isapprox(result.betk_nodist[21:23, 31:33],
            [0.04063917813171917 0.06843246983487516 0.08862506281612659
            0.03684621201600996 0.10352876485995872 0.1255652231824746
            0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-6)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        @test result.ebetk_nodist isa SparseMatrixCSC
        bet_edge_sum = fill(NaN, g.nrows, workspace.grid.ncols)
        for (i, v) in enumerate(sum(result.ebetk_nodist, dims=2))
            bet_edge_sum[g.id_to_grid_coordinate_list[i]] = v
        end
        @test bet_edge_sum[21:23, 31:33] ≈ parent(result.betk_nodist[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as gooda as it can get
        @test isapprox(result.betk_exp50[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test result.betk_one[g.id_to_grid_coordinate_list] ≈
            result.betq[g.id_to_grid_coordinate_list]
        # ebetk_one is wrong here
        @test result.ebetk_one ≈ result.ebetq
    end

    @testset "connected_habitat" begin
        @test result.ch_nodist isa Raster{Float64}
        @test size(result.ch_nodist) == size(g.source_qualities)
        # TODO we need some real tests here
    end
end


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
problem = ConScape.Problem(; graph_measures, connectivity_measure, solver)

@testset "target mosaicing matches original" begin
    # TODO note that this breaks if q weighting is included
    windowed_problem = ConScape.WindowedProblem(problem; 
        buffer=10, centersize=5, threaded=false
    )
    @test collect(ConScape._window_ranges(windowed_problem, rast)) == [
        (1:25, 1:25)   (1:25, 6:30)   (1:25, 11:35)   (1:25, 16:40)   (1:25, 21:45)   (1:25, 26:50)   (1:25, 31:55)   (1:25, 36:59)   (1:25, 41:59)   (1:25, 46:59)   (1:25, 51:59)   (1:25, 56:59)
        (6:30, 1:25)   (6:30, 6:30)   (6:30, 11:35)   (6:30, 16:40)   (6:30, 21:45)   (6:30, 26:50)   (6:30, 31:55)   (6:30, 36:59)   (6:30, 41:59)   (6:30, 46:59)   (6:30, 51:59)   (6:30, 56:59)
        (11:35, 1:25)  (11:35, 6:30)  (11:35, 11:35)  (11:35, 16:40)  (11:35, 21:45)  (11:35, 26:50)  (11:35, 31:55)  (11:35, 36:59)  (11:35, 41:59)  (11:35, 46:59)  (11:35, 51:59)  (11:35, 56:59)
        (16:40, 1:25)  (16:40, 6:30)  (16:40, 11:35)  (16:40, 16:40)  (16:40, 21:45)  (16:40, 26:50)  (16:40, 31:55)  (16:40, 36:59)  (16:40, 41:59)  (16:40, 46:59)  (16:40, 51:59)  (16:40, 56:59)
        (21:44, 1:25)  (21:44, 6:30)  (21:44, 11:35)  (21:44, 16:40)  (21:44, 21:45)  (21:44, 26:50)  (21:44, 31:55)  (21:44, 36:59)  (21:44, 41:59)  (21:44, 46:59)  (21:44, 51:59)  (21:44, 56:59)
        (26:44, 1:25)  (26:44, 6:30)  (26:44, 11:35)  (26:44, 16:40)  (26:44, 21:45)  (26:44, 26:50)  (26:44, 31:55)  (26:44, 36:59)  (26:44, 41:59)  (26:44, 46:59)  (26:44, 51:59)  (26:44, 56:59)
        (31:44, 1:25)  (31:44, 6:30)  (31:44, 11:35)  (31:44, 16:40)  (31:44, 21:45)  (31:44, 26:50)  (31:44, 31:55)  (31:44, 36:59)  (31:44, 41:59)  (31:44, 46:59)  (31:44, 51:59)  (31:44, 56:59)
        (36:44, 1:25)  (36:44, 6:30)  (36:44, 11:35)  (36:44, 16:40)  (36:44, 21:45)  (36:44, 26:50)  (36:44, 31:55)  (36:44, 36:59)  (36:44, 41:59)  (36:44, 46:59)  (36:44, 51:59)  (36:44, 56:59)
        (41:44, 1:25)  (41:44, 6:30)  (41:44, 11:35)  (41:44, 16:40)  (41:44, 21:45)  (41:44, 26:50)  (41:44, 31:55)  (41:44, 36:59)  (41:44, 41:59)  (41:44, 46:59)  (41:44, 51:59)  (41:44, 56:59)
    ]
    test_results = ConScape.solve(windowed_problem, rast; test_windows=true)
    inner_targets = copy(rast.target_qualities)
    replace!(inner_targets, NaN => 0.0)
    # Edge targets are lost with windowing
    inner_targets[1:10, :] .= 0
    inner_targets[:, 1:10] .= 0
    inner_targets[end-9:end, :] .= 0
    inner_targets[:, end-9:end] .= 0
    @test inner_targets == test_results.target_qualities
end

@testset "windowed results approximate non-windowed" begin
    buffer=15
    windowed_problem = ConScape.WindowedProblem(problem; 
        buffer, centersize=5, threaded=false
    )
    mask!(rast; with=rast)
    rast_inner = ConScape._get_window_with_zeroed_buffer(rast, axes(rast), windowed_problem)
    @time wp_result = ConScape.solve(windowed_problem, rast)
    @time p_result = ConScape.solve(problem, rast_inner)
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
    # Use a higher alpha to catch differences
    distance_transformation = x -> exp(-x / 50)
    connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)
    problem = ConScape.Problem(; graph_measures, connectivity_measure, solver)

    kw = (; buffer=10, centersize=5, threaded=false)
    windowed_problem = ConScape.WindowedProblem(problem; kw...)
    windowed_result = ConScape.solve(windowed_problem, rast)

    batch_problem = ConScape.BatchProblem(problem; datapath=tempname(), kw...)
    ConScape.solve(batch_problem, rast)
    batch_result = mosaic(batch_problem; to=rast)
    @test batch_result isa RasterStack

    # BatchProblem can be run as batch jobs for clusters
    # We just need a new path to make sure the result is from a new run
    batch_jobs_problem = ConScape.BatchProblem(problem; 
        datapath=tempname(), joblistpath=tempname(), kw...
    )
    assessment = ConScape.assess(batch_jobs_problem, rast)
    batch_jobs_problem.centersize
    @test assessment.njobs == 39
    @test isfile(batch_jobs_problem.joblistpath)
    ConScape._read_joblist(batch_jobs_problem)

    for job in 1:assessment.njobs
        ConScape.solve(batch_jobs_problem, rast, job)
    end
    batch_jobs_result = mosaic(batch_jobs_problem; to=rast)

    nested_problem = ConScape.BatchProblem(windowed_problem; 
        datapath=tempname(), centersize=(10, 10), threaded=false
    )
    ConScape.assess(nested_problem, rast)
    ConScape.solve(nested_problem, rast)
    nested_result = mosaic(nested_problem; to=rast)
    @test nested_result isa RasterStack

    @test keys(windowed_result) == 
          keys(nested_result) == 
          keys(batch_result) == 
          keys(batch_jobs_result) == Tuple(sort(collect(expected_layers)))

    @test all(permutedims(batch_jobs_result.ch) .=== permutedims(batch_result.ch) .=== windowed_result.ch)
    @test all(permutedims(batch_jobs_result.betk) .=== permutedims(batch_result.betk) .=== windowed_result.betk)

    # TODO: there are some tiny fp differences in the nested result
    @test all(map(nested_result.ch, batch_result.ch) do n, b
        isnan(n) && isnan(b) || isapprox(n, b)
    end)

    # plot(windowed_result)
    # plot(batch_result)
    # plot(batch_jobs_result)
    # plot(nested_result)
end


# Scale Benchmarking...

# windowed_problem_t1 = ConScape.WindowedProblem(problem; 
#     source_radius=10, target_radius=1, threaded=true
# )
# windowed_problem_t2 = ConScape.WindowedProblem(problem; 
#     source_radius=10, target_radius=2, threaded=true
# )
# windowed_problem_t4 = ConScape.WindowedProblem(problem; 
#     source_radius=10, target_radius=4, threaded=true
# )
# windowed_problem_t6 = ConScape.WindowedProblem(problem; 
#     source_radius=10, target_radius=6, threaded=true
# )
# length(ConScape._get_window_ranges(windowed_problem_t1, rast))
# length(ConScape._get_window_ranges(windowed_problem_t2, rast))
# length(ConScape._get_window_ranges(windowed_problem_t4, rast))
# length(ConScape._get_window_ranges(windowed_problem_t6, rast))
# using BenchmarkTools
# @btime ConScape.solve(windowed_problem_t1, rast, verbose=false);
# @btime ConScape.solve(windowed_problem_t2, rast, verbose=false);
# @btime ConScape.solve(windowed_problem_t4, rast, verbose=false);
# @btime ConScape.solve(windowed_problem_t6, rast, verbose=false);
# @profview_allocs ConScape.solve(windowed_problem_t1, rast, verbose=false) sampling=1.0
# @profview_allocs ConScape.solve(windowed_problem_t2, rast, verbose=false) sampling=1.0
# @profview_allocs ConScape.solve(windowed_problem_t4, rast, verbose=false) sampling=1.0
# @profview_allocs ConScape.solve(windowed_problem_t6, rast, verbose=false) sampling=1.0
# @profview 
# res = ConScape.solve(windowed_problem_t1, rast, verbose=false)
# @profview ConScape.solve(windowed_problem_t2, rast, verbose=false)
# @profview ConScape.solve(windowed_problem_t4, rast, verbose=false)
# @profview ConScape.solve(windowed_problem_t6, rast, verbose=false)
# res = ConScape.solve(windowed_problem_t4, rast, verbose=false)
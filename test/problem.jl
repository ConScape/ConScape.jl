nothing
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
# They are only the same for Float32
@test all(Float32.(affinities_asc) .=== Float32.(rast.affinities))
@test all(Float32.(qualities_asc) .=== Float32.(rast.qualities))

graph_measures = graph_measures = (;
    ch=ConScape.ConnectedHabitat(),
    betq=ConScape.BetweennessQweighted(),
    betk=ConScape.BetweennessKweighted(),
    # TODO sens=ConScape.Sensitivity(),
    # ebetq=ConScape.EdgeBetweennessQweighted(),
    # ebetk=ConScape.EdgeBetweennessKweighted(),
    # mkld=ConScape.MeanKullbackLeiblerDivergence(),
    # mlcd=ConScape.MeanLeastCostKullbackLeiblerDivergence(),
    # eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
distance_transformation = (nodist=nothing, one=one, exp50=t -> exp(-t/50))
connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)

expected_layers = (
    :ch_nodist, :ch_one, :ch_exp50, 
    :betq, 
    :betk_nodist, :betk_one, :betk_exp50, 
    # :ebetq, 
    # :ebetk_nodist, :ebetk_one, :ebetk_exp50, 
    # :mkld, 
    # :mlcd,
    # :eigmax_nodist, :eigmax_one, :eigmax_exp50, 
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
    # ConScape.VectorSolver(; threaded=true), # Threading not implemented yet
    # ConScape.LinearSolver(), # TODO: really slow currently
    # ConScape.LinearSolver(; threaded=true),
)

solver = ConScape.VectorSolver()
solver = ConScape.MatrixSolver()

for solver in solvers

@testset "$solver" begin
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem = ConScape.Problem(; 
        graph_measures, connectivity_measure, solver,
    )
    workspace = init(problem, rast; verbose=true)
    pairs(workspace)

    @time result = ConScape.solve!(workspace, problem);

    # @profview result = ConScape.solve(problem, workspace)
    @test size(result.ch_one) == size(rast)
    @test keys(result) == expected_layers
    g = workspace.grid
    sg1 = workspace.subgrids[1]
    # Base.summarysize(workspace) / 1e6
    # ConScape.allocations(problem, size(workspace.B_sparse)).total / 1e6

    # @testset "Test mean_kl_divergence" begin
    #     @test ConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
    #     @test result.mkld[] ≈ 323895.3828183995
    # end

    # @testset "mean_lc_kl_divergence" begin
    #     @test result.mlcd[] ≈ 1.5660600315073947e6
    # end
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
            0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-4)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        # @test result.ebetk_nodist isa SparseMatrixCSC
        # bet_edge_sum = fill(NaN, g.nrows, workspace.grid.ncols)
        # for (i, v) in enumerate(sum(result.ebetk_nodist, dims=2))
            # bet_edge_sum[g.id_to_grid_coordinate_list[i]] = v
        # end
        # @test bet_edge_sum[21:23, 31:33] ≈ parent(result.betk_nodist[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as gooda as it can get
        @test isapprox(result.betk_exp50[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test result.betk_one[sg1.id_to_grid_coordinate_list] ≈ result.betq[sg1.id_to_grid_coordinate_list]
        # @test result.ebetk_one ≈ result.ebetq
    end

    @testset "connected_habitat" begin
        @test result.ch_nodist isa Raster{Float64}
        @test size(result.ch_nodist) == size(g.source_qualities)
        # TODO we need some real tests here
    end
end

end

graph_measures = graph_measures = (;
    ch=ConScape.ConnectedHabitat(),
    betq=ConScape.BetweennessQweighted(),
    betk=ConScape.BetweennessKweighted(),
    # TODO sens=ConScape.Sensitivity(),
    ebetq=ConScape.EdgeBetweennessQweighted(),
    ebetk=ConScape.EdgeBetweennessKweighted(),
    mkld=ConScape.MeanKullbackLeiblerDivergence(),
    mlcd=ConScape.MeanLeastCostKullbackLeiblerDivergence(),
    eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
distance_transformation = (nodist=nothing, one=one, exp50=t -> exp(-t/50))
connectivity_measure = ConScape.ExpectedCost(; θ, distance_transformation)

# All tests for MatrixSolver
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
    # ConScape.VectorSolver(),
    # ConScape.VectorSolver(; threaded=true),
    # ConScape.LinearSolver(),
)
solver = ConScape.MatrixSolver()

for solver in solvers

@testset "$solver complete" begin
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem = ConScape.Problem(; 
        graph_measures, connectivity_measure, solver,
    )
    @time workspace = init(problem, rast);
    Z = copy(workspace.Z)
    @testset "initialised grids are the same" begin
        foreach(propertynames(test_g)) do n
            @test isequal(getproperty(workspace.grid, n), getproperty(test_g, n))
        end
    end

    result = ConScape.solve!(workspace, problem);
    if solver isa ConScape.MatrixSolver
        @test workspace.expected_costs == ConScape.expected_cost(test_grsp)
        # @test workspace.free_energy_distances == ConScape.free_energy_distance(test_grsp)
        @test workspace.Z == test_grsp.Z
        # @test result isa NamedTuple
    end

    @test size(result.ch_one) == size(rast)
    @test keys(result) == expected_layers
    g = workspace.grid

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
        result
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

end
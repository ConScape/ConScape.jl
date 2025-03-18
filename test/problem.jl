nothing
using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL
using ConScape.LinearSolve
using OldConScape

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

θ = 0.1
landscape = "sno_2000"
# The way the ascii is read in is reversed and rotated from what GDAL does
affinities = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "affinities_$landscape.asc")), NaN)); dims=X)
source_qualities = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "qualities_$landscape.asc")), NaN)); dims=X)
source_qualities[(affinities .> 0) .& isnan.(source_qualities)] .= 1e-20
rast = RasterStack((; affinities, source_qualities, target_qualities=source_qualities))

graph_measures = graph_measures = (;
    ch=ConnectedHabitat(),
    betq=Betweenness(QualityWeighted()),
    betk=Betweenness(QualityAndProximityWeighted()),
    ebetq=EdgeBetweenness(QualityWeighted()),
    ebetk=EdgeBetweenness(QualityAndProximityWeighted()),
    mkld=KullbackLeiblerDivergence(),
    # sens=ConScape.Sensitivity(; with_regards_to=ConScape.Cost()),
    # eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
rsp_nodist = RandomisedShortestPath(ConScape.ExpectedCost(); distance_transformation=nothing, theta=θ)
rsp_one = RandomisedShortestPath(ConScape.ExpectedCost(); distance_transformation=one, theta=θ)
rsp_exp_50 = RandomisedShortestPath(ConScape.ExpectedCost(); distance_transformation=ConScape.ExpMinusAlpha(50), theta=θ)
rsp_exp_minus = RandomisedShortestPath(ConScape.ExpectedCost(); distance_transformation=ConScape.ExpMinus(), theta=θ)

solver = ConScape.VectorSolver()

# Precalc
problem = ConScape.Problem(; graph_measures, movement_mode=rsp_exp_minus, solver);
gridinit = init(problem, rast)
subinit = ConScape.init(gridinit, 1)
subgrid1 = ConScape.grid(subinit)
ConScape.sparse_size(gridinit.grid)
ConScape.sparse_size(gridinit.subgrids[1])

affinities_sparse = OldConScape.graph_matrix_from_raster(parent(affinities))
test_g = OldConScape.Grid(size(affinities)...;
    affinities=affinities_sparse,
    qualities=parent(source_qualities)
)
test_grsp = OldConScape.GridRSP(test_g; θ)
qs = [test_grsp.g.source_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list]
qt = [test_grsp.g.target_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list ∩ OldConScape._targetidx_and_nodes(test_g)[1]]
target_1 = ConScape.init(subinit, ConScape.target_ids(subinit)[1])

@test qs == ConScape.source_quality_vector(subinit)
@test qt == ConScape.target_quality_vector(subinit)
@test all(test_g.target_qualities .=== ConScape.target_quality_spatial(subinit))
@test all(test_g.source_qualities .=== ConScape.source_quality_spatial(subinit))
@test test_grsp.W == subinit.W == target_1.W
@test I - test_grsp.W == subinit.IW == target_1.IW
@test test_grsp.Pref == subinit.probability == target_1.probability
@test test_g.costmatrix == subgrid1.costmatrix == target_1.C
@test test_g.costmatrix .* test_grsp.W == subinit.CW == target_1.CW
@test test_g.affinities == subgrid1.affinitymatrix
@test test_g.id_to_grid_coordinate_list == subgrid1.source_ids == ConScape.source_ids(target_1)
@test (test_g.nrows, test_g.ncols) == size(subgrid1) == size(target_1)
@test all(test_g.source_qualities .=== subgrid1.source_quality_spatial)
@test all(test_g.target_qualities .=== subgrid1.target_quality_spatial)

# Dense variables
ec = OldConScape.expected_cost(test_grsp)
fed = OldConScape.free_energy_distance(test_grsp)
sp = OldConScape.survival_probability(test_grsp)
pmp = OldConScape.power_mean_proximity(test_grsp)
Zⁱ = inv.(test_grsp.Z)
Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Zⁱ)) # To prevent Inf*0 later...
QZⁱ = qs .* Zⁱ .* qt'
K = ConScape.ExpMinus().(ec)
M = qs .* K .* qt'
MZⁱ = M .* Zⁱ
betk = zeros(size(test_g))

for i in axes(test_grsp.Z, 2)
    target_i = ConScape.init(subinit, ConScape.target_ids(subinit)[i])
    @test target_i.Z == test_grsp.Z[:, i]
    @test all(isapprox.(target_i.K, K[:, i]))
    @test all(isapprox.(target_i.M, M[:, i]))
    @test all(isapprox.(target_i.MZⁱ, MZⁱ[:, i]))
    @test all(isapprox.(target_i.Zⁱ, Zⁱ[:, i]))
    @test all(isapprox.(target_i.QZⁱ, QZⁱ[:, i]))
    @test all(isapprox.(target_i.QZⁱ, QZⁱ[:, i]))
    @test all(isapprox.(target_i.expected_costs,  ec[:, i]))
    @test all(isapprox.(target_i.free_energy_distances, fed[:, i]; atol=1e-10))
    @test all(isapprox.(target_i.survival_probabilities, sp[:, i]; atol=1e-10))
    @test all(isapprox.(target_i.power_mean_proximities, pmp[:, i]; atol=1e-10))
    @test target_i.qˢ == qs 
    @test target_i.qᵗ == qt[i]
end

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

solvers = (
    ConScape.VectorSolver(),
    # ConScape.VectorSolver(; threaded=true), # Threading not implemented yet
    # ConScape.LinearSolver(), # TODO: really slow currently
    # ConScape.LinearSolver(; threaded=true),
)

for solver in solvers

@testset "$solver" begin
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem_nodist = ConScape.Problem(; graph_measures, movement_mode=rsp_nodist, solver);
    problem_one = ConScape.Problem(; graph_measures, movement_mode=rsp_one, solver);
    problem_exp_50 = ConScape.Problem(; graph_measures, movement_mode=rsp_exp_50, solver);
    problem_exp_minus = ConScape.Problem(; graph_measures, movement_mode=rsp_exp_minus, solver);
    gridinit = init(problem_nodist, rast)
    @time result_nodist = ConScape.solve(problem_nodist, rast);
    @time result_one = ConScape.solve(problem_one, rast);
    @time result_exp_50 = ConScape.solve(problem_exp_50, rast);
    @time result_exp_minus = ConScape.solve(problem_exp_minus, rast);

    @test size(result_nodist.ch) == size(rast)

    @testset "Test mean_kl_divergence" begin
        @test OldConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
        @test result_nodist.mkld[] ≈ 323895.3828183995
    end

    # TODO: make a least-cost section
    # @testset "mean_lc_kl_divergence" begin
    #     @test result.mlcd[] ≈ 1.5660600315073947e6
    # end
    @testset "q-weighted" begin
        @test result_nodist.betq isa Raster
        plot(result_nodist.betq)
        @test isapprox(result_nodist.betq[21:23, 21:23], [
            1930.1334372152335  256.91061166392745 2866.2998374065373
            4911.996715311025  1835.991238248377    720.755518530375
            4641.815380725279  3365.3296878569213   477.1085971945757], atol=1e-3)
    end
    @testset "k-weighted" begin
        @test result_exp_minus.betk isa Raster
        @test isapprox(result_exp_minus.betk[21:23, 31:33], 
            [0.04063917813171917 0.06843246983487516 0.08862506281612659
            0.03684621201600996 0.10352876485995872 0.1255652231824746
            0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-4)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        @test result_nodist.ebetk isa SparseMatrixCSC
        bet_edge_sum = fill(NaN, size(gridinit))
        bet_edge_sum[ConScape.source_ids(gridinit)] .= sum(result_nodist.ebetk, dims=2)
        @test_broken bet_edge_sum[21:23, 31:33] ≈ parent(result_nodist.betk[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as gooda as it can get
        @test isapprox(result_exp_50.betk[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test result_one.betk[ConScape.source_ids(gridinit)] ≈ result_one.betq[ConScape.source_ids(gridinit)]
        @test_broken result_one.ebetk ≈ result_one.ebetq
    end

    @testset "connected_habitat" begin
        @test result_nodist.ch isa Raster{Float64}
        @test size(result_nodist.ch) == size(gridinit)
        # TODO we need some real tests here
    end
end

end

# graph_measures = graph_measures = (;
#     ch=ConnectedHabitat(),
#     betq=Betweenness(QualityWeighted()),
#     betk=Betweenness(QualityAndProximityWeighted()),
#     # TODO sens=ConScape.Sensitivity(),
#     ebetq=ConScape.EdgeBetweenness(QualityWeighted()),
#     ebetk=ConScape.EdgeBetweenness(QualityAndProximityWeighted()),
#     mkld=ConScape.KullbackLeiblerDivergence(),
#     # eigmax=ConScape.EigMax(),
#     # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
# )
# distance_transformation = (nodist=nothing, one=one, exp50=t -> exp(-t/50))
# movement_mode = RandomisedShortestPath(ExpectedCost(); theta=θ, distance_transformation)

# # All tests for MatrixSolver
# expected_layers = (
#     :ch_nodist, :ch_one, :ch_exp50, 
#     :betq, 
#     :betk_nodist, :betk_one, :betk_exp50, 
#     :ebetq, 
#     :ebetk_nodist, :ebetk_one, :ebetk_exp50, 
#     :mkld, 
#     :mlcd,
#     :eigmax_nodist, :eigmax_one, :eigmax_exp50, 
# )

# solvers = (
#     # ConScape.VectorSolver(),
#     # ConScape.VectorSolver(; threaded=true),
#     # ConScape.LinearSolver(),
# )
# solver = ConScape.MatrixSolver()

# for solver in solvers

# @testset "$solver complete" begin
#     println("\n Testing with solver: ", solver)
#     # Basic Problem
#     problem = ConScape.Problem(; 
#         graph_measures, movement_mode, solver,
#     )
#     @time workspace = init(problem, rast);
#     Z = copy(workspace.Z)
#     @testset "initialised grids are the same" begin
#         foreach(propertynames(test_g)) do n
#             @test isequal(getproperty(workspace.grid, n), getproperty(test_g, n))
#         end
#     end

#     result = ConScape.solve(workspace);
#     if solver isa ConScape.MatrixSolver
#         @test workspace.expected_costs == ConScape.expected_cost(test_grsp)
#         # @test workspace.free_energy_distances == ConScape.free_energy_distance(test_grsp)
#         @test workspace.Z == test_grsp.Z
#         # @test result isa NamedTuple
#     end

#     @test size(result.ch_one) == size(rast)
#     @test keys(result) == expected_layers
#     g = workspace.grid

#     @testset "Test mean_kl_divergence" begin
#         @test ConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
#         @test result.mkld[] ≈ 323895.3828183995
#     end

#     @testset "mean_lc_kl_divergence" begin
#         @test result.mlcd[] ≈ 1.5660600315073947e6
#     end
#     @testset "q-weighted" begin
#         @test result.betq isa Raster
#         @test isapprox(result.betq[21:23, 21:23], [
#             1930.1334372152335  256.91061166392745 2866.2998374065373
#             4911.996715311025  1835.991238248377    720.755518530375
#             4641.815380725279  3365.3296878569213   477.1085971945757], atol=1e-3)
#     end
#     @testset "k-weighted" begin
#         @test result.betk_nodist isa Raster
#         bet = ConScape.betweenness_kweighted(test_grsp)
#         result
#         @test isapprox(result.betk_nodist[21:23, 31:33],
#             [0.04063917813171917 0.06843246983487516 0.08862506281612659
#             0.03684621201600996 0.10352876485995872 0.1255652231824746
#             0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-6)

#         # Check that summed edge betweennesses corresponds to node betweennesses:
#         @test result.ebetk_nodist isa SparseMatrixCSC
#         bet_edge_sum = fill(NaN, g.nrows, workspace.grid.ncols)
#         for (i, v) in enumerate(sum(result.ebetk_nodist, dims=2))
#             bet_edge_sum[g.id_to_grid_coordinate_list[i]] = v
#         end
#         @test bet_edge_sum[21:23, 31:33] ≈ parent(result.betk_nodist[21:23, 31:33])

#         # TODO the floating point differnce is more 
#         # significant here, 1e-3 is as gooda as it can get
#         @test isapprox(result.betk_exp50[21:23, 31:33], [
#             980.5828087688377 1307.981162399926 1602.8445739784497
#             826.0710054834001 1883.0940077789735 1935.4450344630702
#             676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

#         @test result.betk_one[g.id_to_grid_coordinate_list] ≈
#             result.betq[g.id_to_grid_coordinate_list]
#         # ebetk_one is wrong here
#         @test result.ebetk_one ≈ result.ebetq
#     end

#     @testset "connected_habitat" begin
#         @test result.ch_nodist isa Raster{Float64}
#         @test size(result.ch_nodist) == size(g.source_qualities)
#         # TODO we need some real tests here
#     end
# end

# end
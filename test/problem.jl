using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
# using LinearSolve
using Rasters, ArchGDAL
using OldConScape

compare(a, b; kw...) = ismissing(a) && ismissing(b) || isnan(a) && isnan(b) || isapprox(a, b; kw...)

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
    fh=FunctionalHabitat(),
    betq=Betweenness(QualityWeighted()),
    betm=MovementFlow(),
    ebetq=EdgeBetweenness(QualityWeighted()),
    ebetm=EdgeBetweenness(QualityAndProximityWeighted()),
    mkld=KullbackLeiblerDivergence(),
    pmp=PowerMeanProximity(),
    sp=SurvivalProbability(),
    ec=ExpectedCost(),
    fed=FreeEnergyDistance(),
    eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)

rsp_nodist = RandomisedShortestPath(ExpectedCost(); theta=θ)
rsp_const_dist = RandomisedShortestPath(ExpectedCost(); distance_transformation=one, theta=θ)
rsp_exp_50 = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinusAlpha(1/50), theta=θ)
rsp_exp_minus = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ)

solver = VectorSolver()

@testset "Compare internals with old conscape" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    test_g = OldConScape.Grid(size(steplikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)
    qs = [test_grsp.g.source_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list]
    qt = [test_grsp.g.target_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list ∩ OldConScape._targetidx_and_nodes(test_g)[1]]

    problem = ConScapeProblem(; 
        measures, movement=rsp_exp_minus, solver, costfunction=MinusLog(),
    );
    gridgraphinit = init(problem, rast)
    connectedgraphinit = init(gridgraphinit, 1)
    connectedgraph1 = ConScape.connectedgraph(connectedgraphinit)
    targetinit1 = init(connectedgraphinit, 1)

    @test qs == ConScape.sourcequality(connectedgraphinit)
    @test qt == ConScape.targetquality(connectedgraphinit)
    @test all(test_g.target_qualities .=== ConScape.targetquality(gridgraphinit))
    @test all(test_g.source_qualities .=== ConScape.sourcequality(gridgraphinit))
    @test test_g.costmatrix == ConScape.stepcost(connectedgraph1) == targetinit1.C
    @test test_g.costmatrix .* test_grsp.W == ConScape.precalculation(connectedgraphinit).CW == targetinit1.CW
    @test test_g.affinities == ConScape.steplikelihood(connectedgraph1)
    @test test_grsp.Pref == ConScape.precalculation(connectedgraphinit).P == targetinit1.P
    @test test_grsp.W == ConScape.precalculation(connectedgraphinit).W == targetinit1.W
    @test LinearAlgebra.I - test_grsp.W == ConScape.precalculation(connectedgraphinit).IW == targetinit1.IW
    @test test_g.id_to_grid_coordinate_list == ConScape.sourceids(connectedgraphinit) == ConScape.sourceids(targetinit1)
    @test (test_g.nrows, test_g.ncols) == size(gridgraphinit)
    @test all(test_g.source_qualities .=== ConScape.sourcequality(gridgraphinit))
    @test all(test_g.target_qualities .=== ConScape.targetquality(gridgraphinit))

    # Dense variables
    ec = OldConScape.expected_cost(test_grsp)
    fed = OldConScape.free_energy_distance(test_grsp)
    sp = OldConScape.survival_probability(test_grsp)
    pmp = OldConScape.power_mean_proximity(test_grsp)
    ch = OldConScape.connected_habitat(test_grsp)
    Zⁱ = inv.(test_grsp.Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Zⁱ)) # To prevent Inf*0 later...
    Q = qs .* qt'
    K = ExpMinus().(ec)
    M = qs .* K .* qt'

    for i in axes(test_grsp.Z, 2)
        target_i = ConScape.init(connectedgraphinit, ConScape.targetids(connectedgraphinit)[i])
        @test target_i.Z == test_grsp.Z[:, i]
        @test all(isapprox.(target_i.Zⁱ, Zⁱ[:, i]))
        @test all(isapprox.(target_i.Q, Q[:, i]))
        @test all(isapprox.(target_i.K, K[:, i]))
        @test all(isapprox.(target_i.M, M[:, i]))
        @test all(isapprox.(ConScape.compute_target(FreeEnergyDistance(), target_i), fed[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute_target(SurvivalProbability(), target_i), sp[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute_target(PowerMeanProximity(), target_i), pmp[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute_target(ExpectedCost(), target_i), ec[:, i]; atol=1e-10))
        @test target_i.qˢ == qs 
        @test target_i.qᵗ == qt[i]
    end

    ec_new = solve!(gridgraphinit, ExpectedCost(), 1)
    btk = OldConScape.betweenness_kweighted(test_grsp);
    btk_new = solve!(gridgraphinit, Betweenness(QualityAndProximityWeighted()))
    @test all(compare.(btk, btk_new))
    btq = OldConScape.betweenness_qweighted(test_grsp);
    btq_new = solve!(gridgraphinit, Betweenness(QualityWeighted()))
    @test all(compare.(btq, btq_new))
    ch = OldConScape.connected_habitat(test_grsp);
    ch_new = solve!(gridgraphinit, FunctionalHabitat())
    @test all(compare.(ch, ch_new))

end

solvers = (
    VectorSolver(),
    # LinearSolver(), # TODO: really slow currently
)

solver = VectorSolver()
for solver in solvers @testset "RSP measures with $solver" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    test_g = OldConScape.Grid(size(steplikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)
    println("\n Testing with solver: ", solver)
    # Basic ConScapeProblem
    problem_nodist = ConScapeProblem(; measures, movement=rsp_nodist, solver);
    problem_const_dist = ConScapeProblem(; measures, movement=rsp_const_dist, solver);
    problem_exp_50 = ConScapeProblem(; measures, movement=rsp_exp_50, solver);
    problem_exp_minus = ConScapeProblem(; measures, movement=rsp_exp_minus, solver);

    gridgraphinit = init(problem_nodist, rast)
    connectedgraphinit = init(gridgraphinit, 1)

    @time result_nodist = solve(problem_nodist, rast);
    @time result_const_dist = solve(problem_const_dist, rast);
    @time result_exp_50 = solve(problem_exp_50, rast);
    @time result_exp_minus = solve(problem_exp_minus, rast);
    @test keys(result_nodist) == keys(measures)
    @test size(result_nodist.fh) == size(rast)

    @testset "Test mean_kl_divergence" begin
        @test OldConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
        result_nodist.mkld[]
        @test result_nodist.mkld[] ≈ 323895.3828183995 atol=1e-2 # This is only slightly off
    end

    @testset "quality weighted" begin
        @test result_nodist.betq isa Raster
        @test isapprox(result_nodist.betq[21:23, 21:23], [
            1930.1334372152335  256.91061166392745 2866.2998374065373
            4911.996715311025  1835.991238248377    720.755518530375
            4641.815380725279  3365.3296878569213   477.1085971945757], atol=1e-3)
    end

    @testset "quality and proximity weighted" begin

        @test result_exp_minus.betm isa Raster
        @test isapprox(result_exp_minus.betm[21:23, 31:33], 
            [0.04063917813171917 0.06843246983487516 0.08862506281612659
            0.03684621201600996 0.10352876485995872 0.1255652231824746
            0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-4)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        connectedgraphinit = init(rsp_exp_minus, rast, 1)
        test_grsp.Z
        old_ebetm = OldConScape.edge_betweenness_kweighted(test_grsp)
        ebetm = solve(EdgeBetweenness(QualityAndProximityWeighted()), rsp_exp_minus, rast, 1)
        lininds = LinearIndices(size(rast))
        @test ebetm isa SparseMatrixCSC
        @test collect(ebetm) ≈ collect(old_ebetm)

        bet_edge_sum = fill(NaN, size(connectedgraphinit))
        bet_edge_sum[ConScape.sourceids(connectedgraphinit)] .= sum(ebetm, dims=2)
        @test bet_edge_sum[21:23, 31:33] ≈ parent(result_nodist.betm[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as good as it can get
        @test isapprox(result_exp_50.betm[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test result_const_dist.betm[ConScape.sourceids(connectedgraphinit)] == 
              result_const_dist.betq[ConScape.sourceids(connectedgraphinit)]
        @test result_const_dist.ebetm ≈ result_const_dist.ebetq

    end

    @testset "connected_habitat" begin
        @test result_exp_minus.fh isa Raster{Float64}
        @test size(result_exp_minus.fh) == size(connectedgraphinit)
        fh = OldConScape.connected_habitat(test_grsp, CartesianIndex((20, 20)))
        # TODO why does this need such a high atol now
        @test all(compare.(result_exp_minus.fh, fh; atol=1e-2))
        # @test cl isa Raster{Float64}
        @test sum(replace(result_exp_minus.fh, NaN => 0.0)) ≈ 109.4795495188798 atol=1e-2
    end

end

end

betweenness_measures = (;
    betu=Betweenness(ConScape.Unweighted()),
    betq=Betweenness(QualityWeighted()),
    betk=Betweenness(ProximityWeighted()),
    betm=Betweenness(QualityAndProximityWeighted()),
)

edge_betweenness_measures = (;
    ebetu=EdgeBetweenness(ConScape.Unweighted()),
    ebetq=EdgeBetweenness(QualityWeighted()),
    ebetk=EdgeBetweenness(ProximityWeighted()),
    ebetm=EdgeBetweenness(QualityAndProximityWeighted()),
)

other_measures = (;
    mkld=KullbackLeiblerDivergence(),
    ch=FunctionalHabitat(),
)

@testset "RSP measures" begin
    # Movement modes
    rsp_ec = RandomisedShortestPath(; 
        proximity_measure=ExpectedCost(), 
        distance_transformation=ExpMinus(),
        theta=1.0, 
    )
    rsp_pmp = RandomisedShortestPath(; 
        proximity_measure=PowerMeanProximity(), 
        distance_transformation=ExpMinus(),
        theta=1.0, 
    )
    @time res_bet_rsp_ec = solve(betweenness_measures, rsp_ec, rast)
    @time res_bet_rsp_pmp = solve(betweenness_measures, rsp_pmp, rast)
    @time res_oth_rsp_ec = solve(other_measures, rsp_ec, rast)
    @time res_oth_rsp_pmp = solve(other_measures, rsp_pmp, rast)
    @time res_ebet_rsp_ec = solve(edge_betweenness_measures, rsp_ec, rast)
    @time res_ebet_rsp_pmp = solve(edge_betweenness_measures, rsp_pmp, rast)
end

@testset "LeastCostPath measures" begin
    lc = LeastCostPath(; distance_transformation=ExpMinusAlpha(2.0),)

    res_bet_lc = solve(betweenness_measures, lc, rast)
    res_oth_lc = solve(other_measures, lc, rast)
    # Not iplemented
    # res_ebet_lc = solve(edge_betweenness_measures, lc, rast)
    # res_sens_lc = solve(sensitivity_measures, lc, rast)

    @testset "LeastCostPath is correlated with RandomisedShortestPath at high theta" begin
        rsp_lc = RSP(; distance_transformation=ExpMinusAlpha(2.0), theta=10.0)
        res_bet_rsp_lc = solve(betweenness_measures, rsp_lc, rast)
        @test cor(collect(skipmissing(res_bet_lc.betq)), collect(skipmissing(res_bet_rsp_lc.betq))) > 0.97
        @test cor(collect(skipmissing(res_bet_lc.betm)), collect(skipmissing(res_bet_rsp_lc.betm))) > 0.97
        # K amd u too broken (by fp over/under-flow ?) to compare
    end

    @testset "mean_lc_kl_divergence" begin
        @test res_oth_lc.mkld[] ≈ 1.5660600315073947e6
    end

    # TODO more tests
end

@testset "RandomWalk measures" begin
    # TODO: test this with alpha other than 1.0
    rw = RandomWalk(; distance_transformation=ExpMinusAlpha(1.0))

    res_bet_rw = solve(betweenness_measures, rw, rast)
    res_oth_rw = solve(other_measures, rw, rast)
    # Not iplemented
    # res_ebet_rw = solve(edge_betweenness_measures, rw, rast)
    # res_sens_rw = solve(sensitivity_measures, rw, rast)

    @testset "RandomWalk is correlated with RandomisedShortestPath at low theta" begin
        rsp_rw = RSP(; distance_transformation=ExpMinusAlpha(1.0), theta=0.000000000001)
        res_bet_rsp_rw = solve(betweenness_measures, rsp_rw, rast)
        @test cor(collect(skipmissing(res_bet_rw.betu)), collect(skipmissing(res_bet_rsp_rw.betu))) > 0.99
        @test cor(collect(skipmissing(res_bet_rw.betq)), collect(skipmissing(res_bet_rsp_rw.betq))) > 0.99
        @test cor(collect(skipmissing(res_bet_rw.betk)), collect(skipmissing(res_bet_rsp_rw.betk))) > 0.97
        # M is less correlated for some reason ?
        @test cor(collect(skipmissing(res_bet_rw.betm)), collect(skipmissing(res_bet_rsp_rw.betm))) > 0.93
    end

    # TODO more tests
end

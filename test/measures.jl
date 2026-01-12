using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
# using LinearSolve
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

compare(a, b; kw...) = ismissing(a) && ismissing(b) || isnan(a) && isnan(b) || isapprox(a, b; kw...)

proximity_measures = (
    fed=FreeEnergyDistance(),
    ec=ExpectedCost(),
    sp=SurvivalProbability(),
    pmp=PowerMeanProximity(),
)

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
    mkld=MeanKullbackLeiblerDivergence(),
    fh=FunctionalHabitat(),
)
     

@testset "Distance and proximity measures" begin
    a = [1.0 1.0
         1.0 1.0]

    g = ConScape.GridGraph(; 
        quality=a,
        cost=a,
        likelihood=a,
        neighbors=ConScape.N4
    )

    problem = ConScapeProblem(;
        measures=proximity_measures,
        movement=RandomisedShortestPath(ExpectedCost(); theta=2.0),
    )

    results = solve(problem, g, 1)
    results.fed

    @test results.fed ≈ [
      0.0       1.34197   1.34197   2.34197
      1.34197   0.0       2.34197   1.34197
      1.34197   2.34197   0.0       1.34197
      2.34197   1.34197   1.34197   0.0     ] atol=1e-4

    @test results.ec ≈ [
      0.0      1.01848  1.01848  2.01848
      1.01848  0.0      2.01848  1.01848
      1.01848  2.01848  0.0      1.01848
      2.01848  1.01848  1.01848  0.0 ] atol=1e-4

    @test results.sp ≈ [
      1.0         0.0682931   0.0682931   0.00924246
      0.0682931   1.0         0.00924246  0.0682931
      0.0682931   0.00924246  1.0         0.0682931
      0.00924246  0.0682931   0.0682931   1.0    ] atol=1e-4

    @test results.pmp ≈ [
      1.0        0.261329   0.261329   0.0961377
      0.261329   1.0        0.0961377  0.261329
      0.261329   0.0961377  1.0        0.261329
      0.0961377  0.261329   0.261329   1.0      ] atol=1e-4
end

measures = (;
    betweenness_measures...,
    edge_betweenness_measures...,
    other_measures...,
    eigmax=ConScape.EigMax(),
)

rsp_nodist = RandomisedShortestPath(ExpectedCost(); theta=θ)
rsp_const_dist = RandomisedShortestPath(ExpectedCost(); distance_transformation=one, theta=θ)
rsp_exp_50 = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinusAlpha(1/50), theta=θ)
rsp_exp_minus = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ)

solvers = (
    VectorSolver(),
    # LinearSolver(), # TODO: really slow currently
)

@testset "RSP measures with $solver" for solver in solvers
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
        result_nodist.mkld[]
        @test result_nodist.mkld[] ≈ 323895.3828183995
   end

    @testset "quality weighted betweenness" begin
        @test result_nodist.betq isa Raster
        @test isapprox(result_nodist.betq[21:23, 21:23], [
            1930.1334372152335  256.91061166392745 2866.2998374065373
            4911.996715311025  1835.991238248377    720.755518530375
            4641.815380725279  3365.3296878569213   477.1085971945757], rtol=1e-7)
    end

    @testset "quality and proximity weighted betweenness" begin
        @test result_exp_minus.betm isa Raster
        @test isapprox(result_exp_minus.betm[21:23, 31:33], 
            [0.04063917813171917 0.06843246983487516 0.08862506281612659
            0.03684621201600996 0.10352876485995872 0.1255652231824746
            0.03190640567704462 0.13832814750469344 0.1961393152256104], rtol=1e-7)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        connectedgraphinit = init(rsp_exp_minus, rast, 1)
        
        lininds = LinearIndices(size(rast))
        @test result_exp_minus.ebetm[1] isa SparseMatrixCSC

        bet_edge_sum = fill(NaN, size(connectedgraphinit))
        bet_edge_sum[ConScape.sourceids(connectedgraphinit)] .= sum(result_nodist.ebetm[1], dims=2)
        @test bet_edge_sum[21:23, 31:33] ≈ parent(result_nodist.betm[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as good as it can get
        @test isapprox(result_exp_50.betm[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], rtol=1e-7)

        @test result_const_dist.betm[ConScape.sourceids(connectedgraphinit)] == 
              result_const_dist.betq[ConScape.sourceids(connectedgraphinit)]
        @test result_const_dist.ebetm ≈ result_const_dist.ebetq

    end

    @testset "Functional Habitat" begin
        @test result_exp_minus.fh isa Raster{Float64}
        @test size(result_exp_minus.fh) == size(connectedgraphinit)
        # TODO why does this need such a high rtol now
        @test sum(replace(result_exp_minus.fh, NaN => 0.0)) ≈ 109.4795495188798 rtol=1e-4
    end
end

@testset "LeastCostPath measures" begin
    lc = LeastCostPath(; distance_transformation=ExpMinus(),)

    res_bet_lc = solve(ConScapeProblem(betweenness_measures, lc), rast)
    res_oth_lc = solve(ConScapeProblem(other_measures, lc), rast)

    # Not iplemented
    # res_ebet_lc = solve(edge_betweenness_measures, lc, rast)

    @testset "LeastCostPath is correlated with RandomisedShortestPath at high theta" begin
        # With theta above 5.0 the output becomes numerically unstable
        rsp_lc = RSP(; distance_transformation=ExpMinus(), theta=5.0)
        res_bet_rsp_lc = solve(ConScapeProblem(betweenness_measures, rsp_lc), rast)
        @test cor(collect(skipmissing(res_bet_lc.betu)), collect(skipmissing(res_bet_rsp_lc.betu))) > 0.92
        @test cor(collect(skipmissing(res_bet_lc.betq)), collect(skipmissing(res_bet_rsp_lc.betq))) > 0.95
        @test cor(collect(skipmissing(res_bet_lc.betk)), collect(skipmissing(res_bet_rsp_lc.betk))) > 0.98
        @test cor(collect(skipmissing(res_bet_lc.betm)), collect(skipmissing(res_bet_rsp_lc.betm))) > 0.98
    end

    @testset "mean_lc_kl_divergence" begin
        @test res_oth_lc.mkld[] ≈ 1.5660600315073947e6
    end
end

@testset "RandomWalk measures" begin
    rw = RandomWalk(; distance_transformation=ExpMinus())

    res_bet_rw = solve(ConScapeProblem(betweenness_measures, rw), rast)
    res_oth_rw = solve(ConScapeProblem(other_measures, rw), rast)

    # Not iplemented
    # res_ebet_rw = solve(edge_betweenness_measures, rw, rast)

    @testset "RandomWalk is correlated with RandomisedShortestPath at low theta" begin
        rsp_rw = RSP(; distance_transformation=ExpMinus(), theta=0.000000000001)
        res_bet_rsp_rw = solve(ConScapeProblem(betweenness_measures, rsp_rw), rast)
        @test cor(collect(skipmissing(res_bet_rw.betu)), collect(skipmissing(res_bet_rsp_rw.betu))) > 0.9999
        @test cor(collect(skipmissing(res_bet_rw.betq)), collect(skipmissing(res_bet_rsp_rw.betq))) > 0.9999
        @test cor(collect(skipmissing(res_bet_rw.betk)), collect(skipmissing(res_bet_rsp_rw.betk))) > 0.96
        @test cor(collect(skipmissing(res_bet_rw.betm)), collect(skipmissing(res_bet_rsp_rw.betm))) > 0.96
    end

    # TODO more tests
end

using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
using Rasters, ArchGDAL
using OldConScape

compare(a, b; kw...) = ismissing(a) && ismissing(b) || isnan(a) && isnan(b) || isapprox(a, b; kw...)
compare(a::SparseMatrixCSC, b::SparseMatrixCSC; kw...) =
    a.colptr == b.colptr && a.rowval == b.rowval && all(isapprox.(a.nzval, b.nzval; kw...))

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
    betu=Betweenness(Unweighted()),
    betq=Betweenness(QualityWeighted()),
    betp=Betweenness(ProximityWeighted()),
    betm=MovementFlow(),
    ebetq=EdgeBetweenness(QualityWeighted()),
    ebetm=EdgeBetweenness(QualityAndProximityWeighted()),
    mkld=MeanKullbackLeiblerDivergence(),
    pmp=PowerMeanProximity(),
    sp=SurvivalProbability(),
    ec=ExpectedCost(),
    fed=FreeEnergyDistance(),
    eigmax=ConScape.EigMax(),
)

rsp_nodist = RandomisedShortestPath(ExpectedCost(); theta=θ)
rsp_const_dist = RandomisedShortestPath(ExpectedCost(); distance_transformation=one, theta=θ)
rsp_exp_50 = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinusAlpha(1/50), theta=θ)
rsp_exp_minus = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ)

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
        measures, movement=rsp_exp_minus, costfunction=MinusLog(),
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
    @test compare(test_g.costmatrix .* test_grsp.W, ConScape.precalculation(connectedgraphinit).CW)
    @test ConScape.precalculation(connectedgraphinit).CW === targetinit1.CW
    @test test_g.affinities == ConScape.steplikelihood(connectedgraph1)
    @test compare(test_grsp.Pref, ConScape.precalculation(connectedgraphinit).P)
    @test ConScape.precalculation(connectedgraphinit).P === targetinit1.P
    @test compare(test_grsp.W, ConScape.precalculation(connectedgraphinit).W)
    @test ConScape.precalculation(connectedgraphinit).W === targetinit1.W
    @test compare(LinearAlgebra.I - test_grsp.W, ConScape.precalculation(connectedgraphinit).IW)
    @test ConScape.precalculation(connectedgraphinit).IW === targetinit1.IW
    @test test_g.id_to_grid_coordinate_list == ConScape.sourceids(connectedgraphinit) == ConScape.sourceids(targetinit1)
    @test (test_g.nrows, test_g.ncols) == size(gridgraphinit)
    @test all(test_g.source_qualities .=== ConScape.sourcequality(gridgraphinit))
    @test all(test_g.target_qualities .=== ConScape.targetquality(gridgraphinit))

    # Dense variables
    Zⁱ = inv.(test_grsp.Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Zⁱ)) # To prevent Inf*0 later...
    Q = qs .* qt'
    K = ExpMinus().(OldConScape.expected_cost(test_grsp))
    M = qs .* K .* qt'

    # Test column-by column match of outputs for each target
    for i in axes(test_grsp.Z, 2)
        target_i = ConScape.init(connectedgraphinit, ConScape.targetids(connectedgraphinit)[i])
        @test all(isapprox.(target_i.Z, test_grsp.Z[:, i]))
        @test all(isapprox.(target_i.Zⁱ, Zⁱ[:, i]))
        @test all(isapprox.(target_i.Q, Q[:, i]))
        @test all(isapprox.(target_i.K, K[:, i]))
        @test all(isapprox.(target_i.M, M[:, i]))
        @test target_i.qˢ == qs 
        @test target_i.qᵗ == qt[i]
    end

    ec_new = solve!(gridgraphinit, ExpectedCost(), 1)

    fh_old = OldConScape.connected_habitat(test_grsp);
    fh_new = solve!(gridgraphinit, FunctionalHabitat())
    @test all(compare.(fh_old, fh_new))

    btk_old = OldConScape.betweenness_kweighted(test_grsp);
    btk_new = solve!(gridgraphinit, Betweenness(QualityAndProximityWeighted()))
    @test all(compare.(btk_old, btk_new))

    btq_old = OldConScape.betweenness_qweighted(test_grsp);
    btq_new = solve!(gridgraphinit, Betweenness(QualityWeighted()))
    @test all(compare.(btq_old, btq_new))

    edge_btk_old = OldConScape.edge_betweenness_kweighted(test_grsp);
    edge_btk_new = solve!(gridgraphinit, EdgeBetweenness(QualityAndProximityWeighted()))
    @test all(compare.(edge_btk_old, edge_btk_new[1]))

    edge_btq_old = OldConScape.edge_betweenness_qweighted(test_grsp);
    edge_btq_new = solve!(gridgraphinit, EdgeBetweenness(QualityWeighted()))
    @test all(compare.(edge_btq_old, edge_btq_new[1]))

    sp_old = OldConScape.survival_probability(test_grsp)
    sp_new = solve!(gridgraphinit, SurvivalProbability())
    @test all(compare.(sp_new[1], sp_old; atol=1e-10))

    fed_old = OldConScape.free_energy_distance(test_grsp)
    fed_new = solve!(gridgraphinit, FreeEnergyDistance())
    @test all(compare.(fed_new[1], fed_old; atol=1e-10))

    pmp_old = OldConScape.power_mean_proximity(test_grsp)
    pmp_new = solve!(gridgraphinit, PowerMeanProximity())
    @test all(compare.(pmp_new[1], pmp_old; atol=1e-10))

    ec_old = OldConScape.expected_cost(test_grsp)
    ec_new = solve!(gridgraphinit, ExpectedCost())
    @test all(compare.(ec_new[1], ec_old; atol=1e-10))
end

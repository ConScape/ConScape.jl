nothing
using ConScape, Test, SparseArrays, LinearAlgebra, LinearSolve
using Rasters, ArchGDAL
using OldConScape

compare(a, b; kw...) = ismissing(a) && ismissing(b) || isnan(a) && isnan(b) || isapprox(a, b; kw...)

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

θ = 0.1
landscape = "sno_2000"
# The way the ascii is read in is reversed and rotated from what GDAL does
movementlikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(movementlikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; movementlikelihood, quality))

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

solver = ConScape.VectorSolver()
# @testset "Compare everything with old conscape" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(movementlikelihood))
    test_g = OldConScape.Grid(size(movementlikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)
    qs = [test_grsp.g.source_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list]
    qt = [test_grsp.g.target_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list ∩ OldConScape._targetidx_and_nodes(test_g)[1]]

    problem = ConScape.Problem(; 
        measures, movement=rsp_exp_minus, solver, costfunction=MinusLog(),
    );
    probleminit = init(problem, rast)
    subgraphinit = init(probleminit, 1)
    subgraph1 = ConScape.connectedgraph(subgraphinit)
    targetinit1 = init(subgraphinit, 1)

    @test qs == ConScape.sourcequality(subgraphinit)
    @test qt == ConScape.targetquality(subgraphinit)
    @test all(test_g.target_qualities .=== ConScape.targetquality(probleminit))
    @test all(test_g.source_qualities .=== ConScape.sourcequality(probleminit))
    @test test_g.costmatrix == subgraph1.transitioncost == targetinit1.C
    @test test_g.costmatrix .* test_grsp.W == subgraphinit.precalculation.CW == targetinit1.CW
    @test test_g.affinities == subgraph1.transitionlikelihood
    @test test_grsp.Pref == subgraphinit.precalculation.P == targetinit1.P
    @test test_grsp.W == subgraphinit.precalculation.W == targetinit1.W
    @test LinearAlgebra.I - test_grsp.W == subgraphinit.precalculation.IW == targetinit1.IW
    @test test_g.id_to_grid_coordinate_list == ConScape.sourceids(subgraphinit) == ConScape.sourceids(targetinit1)
    @test (test_g.nrows, test_g.ncols) == size(probleminit)
    @test all(test_g.source_qualities .=== ConScape.sourcequality(probleminit))
    @test all(test_g.target_qualities .=== ConScape.targetquality(probleminit))

    # Dense variables
    ec = OldConScape.expected_cost(test_grsp)
    fed = OldConScape.free_energy_distance(test_grsp)
    sp = OldConScape.survival_probability(test_grsp)
    pmp = OldConScape.power_mean_proximity(test_grsp)
    ch = OldConScape.connected_habitat(test_grsp)
    Zⁱ = inv.(test_grsp.Z)
    Zⁱ[.!isfinite.(Zⁱ)] .= floatmax(eltype(Zⁱ)) # To prevent Inf*0 later...
    Q = qs .* qt'
    K = ConScape.ExpMinus().(ec)
    M = qs .* K .* qt'

    for i in axes(test_grsp.Z, 2)
        target_i = ConScape.init(subgraphinit, ConScape.targetids(subgraphinit)[i])
        @test target_i.Z == test_grsp.Z[:, i]
        @test all(isapprox.(target_i.Zⁱ, Zⁱ[:, i]))
        @test all(isapprox.(target_i.Q, Q[:, i]))
        @test all(isapprox.(target_i.K, K[:, i]))
        @test all(isapprox.(target_i.M, M[:, i]))
        @test all(isapprox.(ConScape.compute(FreeEnergyDistance(), target_i), fed[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute(SurvivalProbability(), target_i), sp[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute(PowerMeanProximity(), target_i), pmp[:, i]; atol=1e-10))
        @test all(isapprox.(ConScape.compute(ExpectedCost(), target_i), ec[:, i]; atol=1e-10))
        @test target_i.qˢ == qs 
        @test target_i.qᵗ == qt[i]
    end

    ec_new = solve(ExpectedCost(), probleminit, 1)
    btk = OldConScape.betweenness_kweighted(test_grsp);
    btk_new = solve(Betweenness(QualityAndProximityWeighted()), probleminit)
    @test all(compare.(btk, btk_new))
    btq = OldConScape.betweenness_qweighted(test_grsp);
    btq_new = solve(Betweenness(QualityWeighted()), probleminit)
    @test all(compare.(btq, btq_new))
    ch = OldConScape.connected_habitat(test_grsp);
    ch_new = solve(FunctionalHabitat(), probleminit)
    @test all(compare.(ch, ch_new))
end

solvers = (
    ConScape.VectorSolver(),
    # ConScape.LinearSolver(), # TODO: really slow currently
)

# for solver in solvers @testset "$solver" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(movementlikelihood))
    test_g = OldConScape.Grid(size(movementlikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem_nodist = ConScape.Problem(; measures, movement=rsp_nodist, solver);
    problem_const_dist = ConScape.Problem(; measures, movement=rsp_const_dist, solver);
    problem_exp_50 = ConScape.Problem(; measures, movement=rsp_exp_50, solver);
    problem_exp_minus = ConScape.Problem(; measures, movement=rsp_exp_minus, solver);

    probleminit = init(problem_nodist, rast)
    subgraphinit = init(probleminit, 1)

    @time result_nodist = ConScape.solve(problem_nodist, rast);
    @time result_const_dist = ConScape.solve(problem_const_dist, rast);
    @time result_exp_50 = ConScape.solve(problem_exp_50, rast);
    @time result_exp_minus = ConScape.solve(problem_exp_minus, rast);
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

    # @testset "quality and proximity weighted" begin

        @test result_exp_minus.betm isa Raster
        @test isapprox(result_exp_minus.betm[21:23, 31:33], 
            [0.04063917813171917 0.06843246983487516 0.08862506281612659
            0.03684621201600996 0.10352876485995872 0.1255652231824746
            0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-4)

        # Check that summed edge betweennesses corresponds to node betweennesses:
        subgraphinit = init(rsp_exp_minus, rast, 1)
        test_grsp.Z
        old_ebetm = OldConScape.edge_betweenness_kweighted(test_grsp)
        ebetm = solve(EdgeBetweenness(QualityAndProximityWeighted()), rsp_exp_minus, rast, 1)
        lininds = LinearIndices(size(rast))
        @test ebetm isa SparseMatrixCSC
        @test collect(ebetm) ≈ collect(old_ebetm)

        bet_edge_sum = fill(NaN, size(subgraphinit))
        bet_edge_sum[ConScape.sourceids(subgraphinit)] .= sum(ebetm, dims=2)
        @test bet_edge_sum[21:23, 31:33] ≈ parent(result_nodist.betm[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as good as it can get
        @test isapprox(result_exp_50.betm[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test result_const_dist.betm[ConScape.sourceids(subgraphinit)] == 
              result_const_dist.betq[ConScape.sourceids(subgraphinit)]
        @test_broken result_const_dist.ebetm ≈ result_const_dist.ebetq
    end

    @testset "connected_habitat" begin
        @test result_exp_minus.fh isa Raster{Float64}
        @test size(result_exp_minus.fh) == size(subgraphinit)

        fh = OldConScape.connected_habitat(test_grsp, CartesianIndex((20, 20)))
        # TODO why is this so different now
        @test all(compare.(result_exp_minus.fh, fh; atol=1e-2))
        # @test cl isa Raster{Float64}
        @test sum(replace(result_exp_minus.fh, NaN => 0.0)) ≈ 109.4795495188798 atol=1e-2
    end
end

end

# Sensitivity

sensitivity_measures = (;
    sens_cost=SensitivityAnalysis(; wrt=Cost()),
    sens_affinity=SensitivityAnalysis(; wrt=Likelihood()),
    sens_costtoaffinity=SensitivityAnalysis(; wrt=CostToLikelihood()),
    sens_affinitytocost=SensitivityAnalysis(; wrt=LikelihoodToCost()),
    sens_cost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=Cost()),
    sens_affinity_elast=SensitivityAnalysis(; type=Elasticity(), wrt=Likelihood()),
    sens_costtoaffinity_elast=SensitivityAnalysis(; type=Elasticity(), wrt=CostToLikelihood()),
    sens_affinitytocost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=ConScape.LikelihoodToCost()),
)

# Movement modes
# RSP
rsp_pmp = RandomisedShortestPath(; 
    proximity_measure=PowerMeanProximity(), 
    distance_transformation=ExpMinus(),
    costfunction=ConScape.MinusLog(),
    theta=1.0, 
)
rsp_ec = RandomisedShortestPath(; 
    proximity_measure=ExpectedCost(), 
    distance_transformation=ExpMinus(),
    costfunction=ConScape.MinusLog(),
    theta=1.0, 
)

# RSP
@time res_sens_rsp_ec = solve(sensitivity_measures, rsp_ec, rast)
@time res_sens_rsp_pmp = solve(sensitivity_measures, rsp_pmp, rast)

using OldConScape
affinities_sparse = OldConScape.graph_matrix_from_raster(parent(affinities))
test_g = OldConScape.Grid(size(affinities)...;
    affinities=affinities_sparse,
    qualities=parent(source_qualities),
)
test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
old_sens = map((ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)) do distance_transformation
    Dict(wrts .=> map(wrts) do wrt
        old_sens_affinity = OldConScape.sensitivity(test_grsp;
            connectivity_function=OldConScape.expected_cost,
            distance_transformation=OldConScape.ExpMinus(),
            α=0.1,
            wrt,
            landscape_measure=["sum","eigenanalysis"][1],
            unitless=true,
            diagvalue=nothing,
            target_equal_source=true
        )
    end)
end

plot(res_sens_rsp_ec.sens_cost .- old_sens.ec["C"])
heatmap(parent(res_sens_rsp_ec.sens_cost))
heatmap(old_sens.ec["A"])
heatmap(parent(res_sens_rsp_ec.sens_affinity))
heatmap(old_sens.ec["C"])
heatmap(parent(res_sens_rsp_ec.sens_cost))
heatmap(old_sens.ec["C&A=f(C)"])
heatmap(parent(res_sens_rsp_ec.sens_affinitytocost))
heatmap(old_sens.ec["A&C=f(A)"])
heatmap(parent(res_sens_rsp_ec.sens_costtoaffinity))

# OldConScape
affinities_sparse = OldConScape.graph_matrix_from_raster(parent(movementlikelihood))
test_g = OldConScape.Grid(size(movementlikelihood)...;
    affinities=affinities_sparse,
    qualities=parent(quality),
)
test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
old_sens_cost = OldConScape.sensitivity(test_grsp;
    connectivity_function=OldConScape.expected_cost,
    distance_transformation=OldConScape.ExpMinus(),
    α=0.005,
    wrt="C",
    landscape_measure="sum",
    unitless=true, # Elasticities
    diagvalue=nothing,
    target_equal_source=true
)
heatmap(old_sens_cost)
# SensitivityAnalysis
rsp_ec = RandomisedShortestPath(; 
    proximity_measure=ExpectedCost(), 
    distance_transformation=ExpMinusAlpha(0.005),
    costfunction=ConScape.MinusLog(),
    theta=1.0, 
)
sens_cost_elast = SensitivityAnalysis(; 
    type=Elasticity(), wrt=ConScape.Cost()
)
s = solve(sens_cost_elast, rsp_ec, rast, 1)
heatmap(s)
collect(old_sens_cost)
s
old_sens_cost' ≈ s
sum(old_sens_cost, dims=2)
filter(!isnan, vec(s))

# LeastCost
lc = LeastCost(; distance_transformation=ExpMinusAlpha(2.0),)
# RandomWalk
rw = RandomWalk(; distance_transformation=ExpMinusAlpha(1.0))

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

@time res_bet_rsp_ec = solve(betweenness_measures, rsp_ec, rast)
@time res_bet_rsp_pmp = solve(betweenness_measures, rsp_pmp, rast)
@time res_oth_rsp_ec = solve(other_measures, rsp_ec, rast)
@time res_oth_rsp_pmp = solve(other_measures, rsp_pmp, rast)
@time res_ebet_rsp_ec = solve(edge_betweenness_measures, rsp_ec, rast)
@time res_ebet_rsp_pmp = solve(edge_betweenness_measures, rsp_pmp, rast)

res_bet_lc = solve(betweenness_measures, lc, rast)
res_oth_lc = solve(other_measures, lc, rast)
# res_ebet_lc = solve(edge_betweenness_measures, lc, rast)
# res_sens_lc = solve(sensitivity_measures, lc, rast)

@testset "mean_lc_kl_divergence" begin
    @test_broken res_oth_lc.mkld[] ≈ 1.5660600315073947e6
end

# Random Walk
res_bet_rw = solve(betweenness_measures, rw, rast)
res_ebet_rw = solve(edge_betweenness_measures, rw, rast)
res_oth_rw = solve(other_measures, rw, rast)
res_sens_rw = solve(sensitivity_measures, rw, rast)

using Plots
plot(res_sens_rsp_ec; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))
plot(res_sens_rsp_pmp; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))
plot(res_bet_rsp_ec; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))
plot(res_bet_rsp_pmp; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))
plot(res_oth_rsp_pmp.ch; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))
plot(res_oth_rsp_ec.ch; size=(1200, 900))#, layout=(4, 3))#, clims=(0, 2000))

plot(res_bet_rw; size=(1200, 700))
plot(res_oth_rw.ch; size=(1200, 700))

plot(res_bet_lc; size=(1200, 700))
plot(res_oth_lc.ch; size=(1200, 700))
plot(res_sens_rw; size=(1200, 700))

# @testset "eigmax, measure=$proximity_measure" for
#     (proximity_measure, val) in ((ExpectedCost(), 5.576850282179157e6),
#                                      (FreeEnergyDistance(), 3.2799955467465096e6),
#                                      (SurvivalProbability(), 1.3475609129305437e7),
#                                      (PowerMeanProximity(), 3.279995546746518e6))
    proximity_measure, val = (ExpectedCost(), 5.576850282179157e6)
    # proximity_measure, val = (SurvivalProbability(), 1.3475609129305437e7)

    rsp = RSP(; proximity_measure, theta=0.1, distance_transformation=ExpMinus())
    vˡ, λ, vʳ = ConScape.compute(EigMax(), init(init(rsp, rast), 1))

    # Compute the weighted proximity matrix to check results
    M = solve(ConScape.LandscapeMatrix(), rsp, rast, 1)
    λ
    val

    @test_broken λ ≈ val
    @test_broken M * vʳ ≈ vʳ * λ
# end
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
affinities = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
source_qualities = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
source_qualities[(affinities .> 0) .& isnan.(source_qualities)] .= 1e-20
rast = RasterStack((; affinities, source_qualities, target_qualities=source_qualities))

measures = (;
    ch=FunctionalHabitat(),
    betq=Betweenness(QualityWeighted()),
    betm=Betweenness(QualityAndProximityWeighted()),
    ebetq=EdgeBetweenness(QualityWeighted()),
    ebetm=EdgeBetweenness(QualityAndProximityWeighted()),
    mkld=KullbackLeiblerDivergence(),
    pmp=PowerMeanProximity(),
    sp=SurvivalProbability(),
    ec=ExpectedCost(),
    fed=FreeEnergyDistance(),
    # eigmax=ConScape.EigMax(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)

rsp_nodist = RandomisedShortestPath(ExpectedCost(); theta=θ)
rsp_one = RandomisedShortestPath(ExpectedCost(); distance_transformation=one, theta=θ)
rsp_exp_50 = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinusAlpha(50), theta=θ)
rsp_exp_minus = RandomisedShortestPath(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ)

solver = ConScape.VectorSolver()
@testset "Compare everything with old conscape" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(affinities))
    test_g = OldConScape.Grid(size(affinities)...;
        affinities=affinities_sparse,
        qualities=parent(source_qualities)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)

    problem = ConScape.Problem(; measures, movement=rsp_exp_minus, solver);
    multigridinit = init(problem, rast)
    gridinit = init(multigridinit, 1)
    subgrid1 = ConScape.grid(gridinit)
    qs = [test_grsp.g.source_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list]
    qt = [test_grsp.g.target_qualities[i] for i in test_grsp.g.id_to_grid_coordinate_list ∩ OldConScape._targetidx_and_nodes(test_g)[1]]
    target_1 = init(gridinit, 1)

    @test qs == ConScape.source_quality_vector(gridinit)
    @test qt == ConScape.target_quality_vector(gridinit)
    @test all(test_g.target_qualities .=== ConScape.target_quality_spatial(gridinit))
    @test all(test_g.source_qualities .=== ConScape.source_quality_spatial(gridinit))
    @test test_g.costmatrix == subgrid1.costmatrix == target_1.C
    @test test_g.costmatrix .* test_grsp.W == gridinit.precalculation.CW == target_1.CW
    @test test_g.affinities == subgrid1.affinitymatrix
    @test test_grsp.Pref == gridinit.precalculation.P == target_1.P
    @test test_grsp.W == gridinit.precalculation.W == target_1.W
    @test I - test_grsp.W == gridinit.precalculation.IW == target_1.IW
    @test test_g.id_to_grid_coordinate_list == subgrid1.source_ids == ConScape.source_ids(target_1)
    @test (test_g.nrows, test_g.ncols) == size(subgrid1) == size(target_1)
    @test all(test_g.source_qualities .=== subgrid1.source_quality_spatial)
    @test all(test_g.target_qualities .=== subgrid1.target_quality_spatial)

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
        target_i = ConScape.init(gridinit, ConScape.target_ids(gridinit)[i])
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

    ec_new = solve(ExpectedCost(), multigridinit)
    btk = OldConScape.betweenness_kweighted(test_grsp);
    btk_new = solve(Betweenness(QualityAndProximityWeighted()), multigridinit)
    @test all(compare.(btk, btk_new))
    btq = OldConScape.betweenness_qweighted(test_grsp);
    btq_new = solve(Betweenness(QualityWeighted()), multigridinit)
    @test all(compare.(btq, btq_new))
    ch = OldConScape.connected_habitat(test_grsp);
    ch_new = solve(FunctionalHabitat(), multigridinit)
    @test all(compare.(ch, ch_new))
end


solvers = (
    ConScape.VectorSolver(),
    # ConScape.LinearSolver(), # TODO: really slow currently
)

for solver in solvers @testset "$solver" begin
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(affinities))
    test_g = OldConScape.Grid(size(affinities)...;
        affinities=affinities_sparse,
        qualities=parent(source_qualities)
    )
    test_grsp = OldConScape.GridRSP(test_g; θ)
    println("\n Testing with solver: ", solver)
    # Basic Problem
    problem_nodist = ConScape.Problem(; measures, movement=rsp_nodist, solver);
    problem_one = ConScape.Problem(; measures, movement=rsp_one, solver);
    problem_exp_50 = ConScape.Problem(; measures, movement=rsp_exp_50, solver);
    problem_exp_minus = ConScape.Problem(; measures, movement=rsp_exp_minus, solver);
    gridinit = init(problem_nodist, rast)

    @time result_nodist = ConScape.solve(problem_nodist, rast);
    @time result_one = ConScape.solve(problem_one, rast);
    @time result_exp_50 = ConScape.solve(problem_exp_50, rast);
    @time result_exp_minus = ConScape.solve(problem_exp_minus, rast);
    @test keys(result_nodist) == keys(measures)

    @test size(result_nodist.ch) == size(rast)

    @testset "Test mean_kl_divergence" begin
        @test OldConScape.mean_kl_divergence(test_grsp) ≈ 323895.3828183995
        result_nodist.mkld[]
        @test result_nodist.mkld[] ≈ 323895.3828183995
    end

    # TODO: make a least-cost section
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
        @test result_nodist.ebetm isa Matrix
        bet_edge_sum = fill(NaN, size(gridinit))
        bet_edge_sum[ConScape.source_ids(gridinit)] .= sum(result_nodist.ebetm, dims=2)
        @test_broken bet_edge_sum[21:23, 31:33] ≈ parent(result_nodist.betm[21:23, 31:33])

        # TODO the floating point differnce is more 
        # significant here, 1e-3 is as gooda as it can get
        @test isapprox(result_exp_50.betm[21:23, 31:33], [
            980.5828087688377 1307.981162399926 1602.8445739784497
            826.0710054834001 1883.0940077789735 1935.4450344630702
            676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

        @test_broken result_one.betm[ConScape.source_ids(gridinit)] ≈ 
            result_one.betq[ConScape.source_ids(gridinit)]
        @test_broken result_one.ebetm ≈ result_one.ebetq
    end

    @testset "connected_habitat" begin
        @test result_exp_minus.ch isa Raster{Float64}
        @test size(result_exp_minus.ch) == size(gridinit)

        ch = OldConScape.connected_habitat(test_grsp, CartesianIndex((20, 20)))
        # TODO why is this so different now
        @test all(compare.(result_exp_minus.ch, ch; atol=1e-2))
        # @test cl isa Raster{Float64}
        @test sum(replace(result_exp_minus.ch, NaN => 0.0)) ≈ 109.4795495188798 atol=1e-2
    end
end
end

# Measures

sensitivity_measures = (;
    sens_cost=ConScape.Sensitivity(; context=ConScape.Cost()),
    sens_affinity=ConScape.Sensitivity(; context=ConScape.Affinity()),
    sens_costtoaffinity=ConScape.Sensitivity(; context=ConScape.CostToAffinity()),
    sens_affinitytocost=ConScape.Sensitivity(; context=ConScape.AffinityToCost()),
    sens_cost_prop=ConScape.Sensitivity(; change=ConScape.ProportionalChange(), context=ConScape.Cost()),
    sens_affinity_prop=ConScape.Sensitivity(; change=ConScape.ProportionalChange(), context=ConScape.Affinity()),
    sens_costtoaffinity_prop=ConScape.Sensitivity(; change=ConScape.ProportionalChange(), context=ConScape.CostToAffinity()),
    sens_affinitytocost_prop=ConScape.Sensitivity(; change=ConScape.ProportionalChange(), context=ConScape.AffinityToCost()),
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
    mkld=KullbackLeiblerDivergence(),
    ch=FunctionalHabitat(),
)

# Movement modes
# RSP
rsp_pmp = RandomisedShortestPath(; 
    proximity_measure=PowerMeanProximity(), 
    distance_transformation=ExpMinusAlpha(1.0),
    costfunction=ConScape.MinusLog(),
    theta=0.01, 
)
rsp_ec = RandomisedShortestPath(; 
    proximity_measure=ExpectedCost(), 
    distance_transformation=ExpMinusAlpha(1),
    costfunction=ConScape.MinusLog(),
    theta=0.01, 
)
# LeastCost
lc = LeastCost(; distance_transformation=ExpMinusAlpha(2.0),)
# RandomWalk
rw = RandomWalk(; distance_transformation=ExpMinusAlpha(1.0))

# RSP
@time res_sens_rsp_ec = solve(sensitivity_measures, rsp_ec, rast)
@time res_sens_rsp_pmp = solve(sensitivity_measures, rsp_pmp, rast)
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
# res_sens_rw = solve(sensitivity_measures, rw, rast)

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

# using Plots
# for i in 1500:2000
#     @show i
#     v = rw.ec[:, i]
#     any(>(0), v) || continue
#     display(Plots.heatmap(reshape(v, size(rast)); size=(1200, 700)))
# end
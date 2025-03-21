
using ConScape, Test, SparseArrays, OldConScape
using Rasters, ArchGDAL, Plots

# include("problem.jl")

# TODO reorganise this into separate files

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

# @testset "sno_2000 Rasters" begin
#     landscape = "sno_2000"
#     θ = 0.1

#     # The way the ascii is read in is reversed and rotated from what GDAL does
#     affinity_raster = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "affinities_$landscape.asc")), NaN)); dims=X)
#     affinitymatrix = ConScape.graph_matrix_from_raster(affinity_raster)
#     @test Float32.(affinitymatrix[1000:1002, 1000:1002]) == Float32.([
#         0.0               0.00031508895477488 0.0
#         0.133336775193571 0.0                 0.00119533310704962
#         0.0               0.00031508895477488 0.0])

#     qualities = reverse(rotr90(replace_missing(Raster(joinpath(datadir, "qualities_$landscape.asc")), NaN)); dims=X)
#     @test dims(qualities) == dims(affinity_raster)

#     qualities[(affinity_raster .> 0) .& isnan.(qualities)] .= 1e-20

#     g = ConScape.Grid(size(affinity_raster);
#         affinitymatrix,
#         source_qualities=qualities
#     )
#     @test dims(g) === dims(qualities)

#     # @testset "Rasters are returned" begin
#         # @test ConScape.indegrees(g) isa Raster
#         # @test ConScape.outdegrees(g) isa Raster
#         # @test Raster(ones(length(g.id_to_grid_coordinate_list)), g) isa Raster
#     # end

#     grsp = ConScape.GridRSP(g, θ=θ)
#     @test dims(grsp) === dims(affinity_raster)

#     @testset "Test mean_kl_divergence" begin
#         @test_broken ConScape.mean_kl_divergence(grsp) ≈ 323895.3828183995
#     end

#     @testset "mean_lc_kl_divergence" begin
#         @test_broken ConScape.mean_lc_kl_divergence(grsp) ≈ 1.5660600315073947e6
#     end

#     @testset "test adjacency creation with $nn neighbors, $w weighting and $mt" for
#         nn in (ConScape.N4, ConScape.N8),
#             w in (ConScape.TargetWeight, ConScape.AverageWeight),
#                 mt in (ConScape.AffinityMatrix, ConScape.CostMatrix)
#                     # No need to test this on sno_100 and doesn't deepend on θ
#                     # FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
#                     @test ConScape.graph_matrix_from_raster(
#                         affinity_raster,
#                         neighbors=nn,
#                         weight=w,
#                         matrix_type=mt) isa ConScape.SparseMatrixCSC
#     end

#     @testset "Test betweenness" begin
#         @testset "q-weighted" begin
#             bet = ConScape.betweenness_qweighted(grsp)
#             @test bet isa Raster
#             @test isapprox(bet[21:23, 21:23], [
#                 1930.1334372152335  256.91061166392745 2866.2998374065373
#                 4911.996715311025  1835.991238248377    720.755518530375
#                 4641.815380725279  3365.3296878569213   477.1085971945757], atol=1e-3)
#         end

#         @testset "k-weighted" begin
#             bet = compute(Betweenness(QualityAndProximityWeighted()), grsp))(grsp, diagvalue=1.)
#             @test bet isa Raster
#             @test isapprox(bet[21:23, 31:33], [
#                 0.04063917813171917 0.06843246983487516 0.08862506281612659
#                 0.03684621201600996 0.10352876485995872 0.1255652231824746
#                 0.03190640567704462 0.13832814750469344 0.1961393152256104], atol=1e-6)

#             # Check that summed edge betweennesses corresponds to node betweennesses:
#             bet_edge = ConScape.edge_betweenness_kweighted(grsp, diagvalue=1.)
#             @test bet_edge isa SparseMatrixCSC
#             bet_edge_sum = fill(NaN, grsp.g.nrows, grsp.g.ncols)
#             for (i, v) in enumerate(sum(bet_edge,dims=2))
#                 bet_edge_sum[grsp.g.id_to_grid_coordinate_list[i]] = v
#             end
#             @test bet_edge_sum[21:23, 31:33] ≈ bet[21:23, 31:33]

#             # This is a regression test based on values that we currently believe to be correct
#             bet = ConScape.betweenness_kweighted(grsp, distance_transformation=t -> exp(-t/50))
#             # TODO the floating point differnce is more 
#             # significant here, 1e-3 is as gooda as it can get
#             @test isapprox(bet[21:23, 31:33], [
#                 980.5828087688377 1307.981162399926 1602.8445739784497
#                 826.0710054834001 1883.0940077789735 1935.4450344630702
#                 676.9212075214159 2228.2700913772774 2884.0409495023364], atol=1e-3)

#             @test ConScape.betweenness_kweighted(grsp, distance_transformation=one)[g.id_to_grid_coordinate_list] ≈
#                 ConScape.betweenness_qweighted(grsp)[g.id_to_grid_coordinate_list]

#             @test ConScape.edge_betweenness_kweighted(grsp, distance_transformation=one) ≈
#                 ConScape.edge_betweenness_qweighted(grsp)
#         end

#     end

#     @testset "connected_habitat" begin
#         ch = ConScape.connected_habitat(grsp)
#         @test ch isa Raster{Float64}
#         @test size(ch) == size(grsp.g.source_qualities)

#         cl = ConScape.connected_habitat(grsp, CartesianIndex((20,20)))
#         @test cl isa Raster{Float64}
#         @test sum(replace(cl, NaN => 0.0)) ≈ 109.4795495188798
#     end

#     @testset "Show methods" begin
#         b = IOBuffer()
#         show(b, "text/plain", g)
#         @test occursin("Grid", String(take!(b)))

#         b = IOBuffer()
#         show(b, "text/plain", grsp)
#         @test occursin("GridRSP", String(take!(b)))
#     end
# end

# Tests with non-standard landcapes
@testset "graph splitting" begin
    l1 = [1/4 0 1/4 1/4
          1/4 0 1/4 1/4
          1/4 0 1/4 1/4
          1/4 0 1/4 1/4]

    l2 = [0   0 1/4 1/4
          0   0 1/4 1/4
          0   0 1/4 1/4
          0   0 1/4 1/4]

    g1 = ConScape.Grid(size(l1), affinitymatrix=ConScape.graph_matrix_from_raster(l1))
    g2 = ConScape.Grid(size(l2), affinitymatrix=ConScape.graph_matrix_from_raster(l2))
    sgs1 = ConScape.split_subgraphs(g1)
    sgs2 = ConScape.split_subgraphs(g2)
    @test length(sgs1) == 2
    @test length(sgs2) == 1

    @test !ConScape.is_strongly_connected(g1)
    @test ConScape.is_strongly_connected(sgs1[1])
    @test ConScape.is_strongly_connected(sgs1[2])
    @test !ConScape.is_strongly_connected(g2) # Why not?
    @test ConScape.is_strongly_connected(sgs2[1])

    sgs1[1].costmatrix == sgs1[1].costmatrix
    sgs1[1].affinitymatrix == sgs1[1].affinitymatrix

    g1.costmatrix
    g2.costmatrix
end

# @testset "least cost distance" begin
    r = [1/4 0 1/2 1/4
         1/4 0 1/2 1/4
         1/4 0 1/2 1/4
         1/4 0 1/2 1/4]

    a = ConScape.graph_matrix_from_raster(r, neighbors=ConScape.N4)
    c = copy(a)
    c.nzval .= 1/2

    # @testset "_cost: $_cost, op: $op, prune: $prune" for
        # (_cost, op) in ((ConScape.MinusLog(), <), (c, ==)),
            # prune in (true, false)

        g = ConScape.Grid(size(r), affinitymatrix=a, costmatrix=c)
        lc = solve(ExpectedCost(), LeastCost(), g)
        @test prune || all(isinf, lc[1:8, 9:16])
        # since (4, 3) -> (4, 4) has higher affinity than (3, 4) -> (4, 4), i.e. lower cost
        # when costs=MinusLog() and identical affinities and costs when using the cost matrix c
        # if prune
            # pruned landscape has size (4, 2)
            # @test op(lc[(1 - 1)*4 + 4, 8], lc[(2 - 1)*4 + 3, 8])
        else
            # full landscape has size (4, 4)
            @test op(lc[(3 - 1)*4 + 4, 16], lc[(4 - 1)*4 + 3, 16])
        end
    end

end

# @testset "Distances and proximities" begin
    l = [1 1
         1 1]

    a = ConScape.graph_matrix_from_raster(l, neighbors=ConScape.N4)

    c = ConScape.graph_matrix_from_raster(l;
        neighbors=ConScape.N4,
        matrix_type=ConScape.CostMatrix
    )

    @testset "check shapes of affinity and cost matrices" begin
        @test_throws ArgumentError("grid size (2, 2) is incompatible with size of affinity matrix (3, 3)") ConScape.Grid(size(l),
            affinitymatrix=a[1:end-1, 1:end-1],
            costmatrix=c
        )

        @test_throws ArgumentError("grid size (2, 2) is incompatible with size of cost matrix (3, 3)") ConScape.Grid(
            size(l),
            affinitymatrix=a,
            costmatrix=c[1:end-1, 1:end-1]
        )
    end

    grid = ConScape.Grid(size(l),
        affinitymatrix=a,
        costmatrix=c
    )
    measures = (
        fed=FreeEnergyDistance(),
        ec=ExpectedCost(),
        sp=SurvivalProbability(),
    )
     
    problem = ConScape.Problem(;
        measures,
        movement_mode=RandomisedShortestPath(ExpectedCost(); theta=2.0),
    )

    results = solve(problem, grid)

    @test free_energy_grsp ≈ [
      0.0       1.34197   1.34197   2.34197
      1.34197   0.0       2.34197   1.34197
      1.34197   2.34197   0.0       1.34197
      2.34197   1.34197   1.34197   0.0     ] atol=1e-4
    @test result.fed ≈ free_energy_grsp

    @test excepted_cost_grsp ≈ [
      0.0      1.01848  1.01848  2.01848
      1.01848  0.0      2.01848  1.01848
      1.01848  2.01848  0.0      1.01848
      2.01848  1.01848  1.01848  0.0 ] atol=1e-4
    @test results.ec ≈ excepted_cost_grsp

    survival_probability_grsp = ConScape.survival_probability(grsp)
    @test survival_probability_grsp ≈ [
      1.0         0.0682931   0.0682931   0.00924246
      0.0682931   1.0         0.00924246  0.0682931
      0.0682931   0.00924246  1.0         0.0682931
      0.00924246  0.0682931   0.0682931   1.0    ] atol=1e-4
    @test results.sp ≈ survival_probability_grsp

    @test power_mean_proximity_grsp ≈ [
      1.0        0.261329   0.261329   0.0961377
      0.261329   1.0        0.0961377  0.261329
      0.261329   0.0961377  1.0        0.261329
      0.0961377  0.261329   0.261329   1.0      ] atol=1e-4
    @test results.pmp ≈ power_mean_proximity_grsp
end

@testset "custom scaling function in k-weighted betweenness" begin
    l = rand(4, 4)
    q = rand(4, 4)

    g = ConScape.Grid(size(l); affinitymatrix=ConScape.graph_matrix_from_raster(l))

    bet = Betweenness(QualityAndProximityWeighted())
    rsp_me = RandomisedShortestPath(; theta=0.2, distance_transformation=ExpMinus())
    rsp_f = RandomisedShortestPath(; theta=0.2, distance_transformation=t -> exp(-t))
    @test solve(bet, rsp_me, g) == solve(bet, rsp_f, g)
end

@testset "least cost kl divergence" begin

    C = sparse([0.0 1 0 0 0
                1.0 0 9 3 0
                0.0 9 0 0 5
                0.0 3 0 0 4
                0.0 0 5 4 0])

    A = sparse([0.0  5  0  0  0
                3.0  0 10 17  0
                0.0  3  0  0  2
                0.0  6  0  0 19
                0.0  0 14 17  0])

    Pref = ConScape._Pref(A)

    @test hcat([ConScape.least_cost_kl_divergence(C, Pref, i) for i in 1:5]...) ≈
        [0.0                0.0                 1.0986122886681098  0.5679840376059393  0.8424208833076996
         2.3025850929940455 0.0                 1.0986122886681098  0.5679840376059393  0.8424208833076996
         2.813410716760036  0.5108256237659905  0.0                 1.5170645923030852  0.916290731874155
         3.7297014486341915 1.4271163556401458  1.069366720571648   0.0                 0.2744368457017603
         4.330475309063122  2.027890216069076   0.7949298748698876  0.6007738604289302  0.0               ]

    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3, 2))
    @test compute(KullbackLeiblerDivergence(), LeastCost(), g, (25, 50))[10, 10] ≈ 80.63375074079197
    @test ConScape.least_cost_kl_divergence(grsp, (25,50))[10,10] ≈ 80.63375074079197
end

# FIXME! Computation is currently very slow so we have to use a reduced landscape
@testset "Criticality" begin
    m, n = 10, 15
    g = ConScape.permeable_wall_sim(m, n, corridorwidths=(2,2),
        # Qualities decrease by row
        qualitymatrix=copy(reshape(collect(m*n:-1:1), n, m)')
    )
    rsp = RandomisedShortestPath(; theta=0.2)
    crt = conpute(Criticality(), rsp, g)
    @test sum(t -> isnan(t) ? 0.0 : t, crt .< -1e-5) == 0
end

# @testset "pass cost matrix instead of function" begin
    m, n = 10, 15

    _g = ConScape.permeable_wall_sim(m, n, corridorwidths=(2,2),
        # Qualities decrease by row
        qualities=copy(reshape(collect(m*n:-1:1), n, m)'))

    g = ConScape.Grid(m, n,
        affinitymatrix=_g.affinitymatrix,
        source_qualities=_g.source_quality_spatial,
        costfunction=ConScape.MinusLog())
    rsp = RandomisedShortestPath(; theta=0.2)

    g_with_costs = ConScape.Grid(m, n,
        affinitymatrix=_g.affinitymatrix,
        source_qualities=_g.source_quality_spatial,
        costmatrix=ConScape.mapnz(ConScape.MinusLog(), _g.affinitymatrix)
    )
    rsp = RandomisedShortestPath(; theta=0.2)

    # @test 
    g_with_costs.costfunction
    #  === nothing

    betq = Betweenness(QualityWeighted())
    @test solve(betq, rsp, g) == ConScape.solve(betq, rsp, g_with_costs)

    # For betweenness_kweighted and connected_habitat we should have exact match between the two
    # methods of passing the costs
    for f in (:betweenness_kweighted, :connected_habitat)
        @test_throws ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function.") getfield(ConScape, f)(grsp_with_costs)

        @test getfield(ConScape, f)(grsp_with_costs, connectivity_function=ConScape.survival_probability) isa AbstractMatrix
        @test getfield(ConScape, f)(grsp, distance_transformation=ConScape.ExpMinus()) == getfield(ConScape, f)(grsp_with_costs, distance_transformation=ConScape.ExpMinus())

        @test getfield(ConScape, f)(grsp, distance_transformation=ConScape.Inv(), diagvalue=1.0) == getfield(ConScape, f)(grsp_with_costs, distance_transformation=ConScape.Inv(), diagvalue=1.0)
    end

    # ...this is not the case for criticality because we don't set the affinity to zero but a very small
    # number. Therefore, the costs will get updated when a cost function is suppled but not when cost
    # matrix is supplied. The difference appear to be small, though, so we can test with ≈
    for f in (:criticality,)
        @test_throws ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function.") getfield(ConScape, f)(grsp_with_costs)
        @test getfield(ConScape, f)(grsp, distance_transformation=ConScape.ExpMinus()) ≈ getfield(ConScape, f)(grsp_with_costs, distance_transformation=ConScape.ExpMinus())
        @test getfield(ConScape, f)(grsp, distance_transformation=ConScape.Inv(), diagvalue=1.0) ≈ getfield(ConScape, f)(grsp_with_costs, distance_transformation=ConScape.Inv(), diagvalue=1.0)
    end
end

@testset "Cost functions" begin
    l = rand(4, 4)
    affinitymatrix = ConScape.graph_matrix_from_raster(l)

    for c in [ConScape.MinusLog(),
              ConScape.ExpMinus(),
              ConScape.Inv(),
              ConScape.OddsAgainst(),
              ConScape.OddsFor()]

        g = ConScape.Grid(size(l);
            affinitymatrix=affinitymatrix,
            costfunction=c
        )

        h_c = init(RandomisedShortestPath(; theta=0.2), grid)
        @test h_c isa ConScape.MultiGridInitialisation{<:ConScape.Problem{<:RandomisedShortestPath}}
    end

    affinities[1,2] = 1.1 # Causes negative cost for C[1,2] when costs=MinusLog
    # Broken check
    # @test_throws ArgumentError ConScape.Grid(
    #     size(l)...,
    #     affinities=affinities,
    #     costs=ConScape.MinusLog()) # should raise error, as C[1,2]<0
end

# @testset "Avoid NaNs when Z has tiny values" begin
    mov_prob = reverse(rotr90(Raster(joinpath(datadir, "mov_prob_1000.asc"); missingval=NaN)); dims=X);

    q = zeros(size(mov_prob))
    q[60,70]   = 1
    q[50, 105] = 1
    g = ConScape.Grid(size(mov_prob),
        affinitymatrix=ConScape.graph_matrix_from_raster(mov_prob),
        qualities=q,
        costfunction=ConScape.MinusLog()
    );
    rsp = RandomisedShortestPath(; theta=2.5)
    betw = solve(Betweenness(QualityAndProximityWeighted()), rsp, g)
    g_old = OldConScape.Grid(size(mov_prob)...;
        affinities=ConScape.graph_matrix_from_raster(mov_prob),
        qualities=q,
        costs=OldConScape.MinusLog()
    );
    betw = OldConScape.betweenness_kweighted(OldConScape.GridRSP(g_old; θ=2.5))
    # @test 
    betw[58:60, 78:80]
     ≈ [
        0.397426   0.170278   0.348822
        1.42686    1.65378    1.419
        0.0554379  0.0192699  0.185261] rtol=1e-3
end

# @testset "Avoid overflow in k-weighted betweenness" begin
    mov_prob, meta_p = ConScape.readasc(joinpath(datadir, "mov_prob_200.asc"))
    hab_qual, meta_q = ConScape.readasc(joinpath(datadir, "hab_qual_200.asc"))

    # FIXME! We'd have to handle this somehow in the library
    @test_broken isnan.(mov_prob) == isnan.(hab_qual)
    non_matches = findall(xor.(isnan.(mov_prob), isnan.(hab_qual)))
    mov_prob[non_matches] .= 1e-20
    hab_qual[non_matches] .= 1e-20

    g = ConScape.Grid(size(mov_prob)...,
        affinitymatrix=ConScape.graph_matrix_from_raster(mov_prob),
        qualities=hab_qual,
        costfunction=ConScape.MinusLog()
    )

    g_coarse = ConScape.Grid(size(mov_prob);
        affinitymatrix=ConScape.graph_matrix_from_raster(mov_prob),
        source_qualities=hab_qual,
        target_qualities=ConScape.coarse_graining(g, 200),
        costfunction=ConScape.MinusLog()
    )

    h_coarse = ConScape.GridRSP(g_coarse, θ=1.0)

    kbetw = @time solve(Betweenness(QualityAndProximityWeighted()), h_coarse, distance_transformation=x -> exp(-x/100))
    @test count(!isnan, kbetw) == 128234
end

@testset "Test that cost edges are contained in the affinity edges" begin
    @test_throws ArgumentError("cost graph contains edges not present in the affinity graph") ConScape.Grid((2, 2);
        affinitymatrix=sparse(
            [3, 4, 1, 4, 2, 3],
            [1, 2, 3, 3, 4, 4],
            [1.0, 1, 1, 1, 1, 1]),
        costmatrix=sparse(
            [2, 3, 1, 4, 1, 4, 2, 3],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1.0, 1, 1, 1, 1, 1, 1, 1]), 
        check=true
    )
end
# 
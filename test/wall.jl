using ConScape, Test, SparseArrays

@testset "wall full" begin
    θ = 0.2
    # Create the same landscape in Julia
    g = ConScape.permeable_wall_sim(30, 60; corridorwidths=(3,2),
        # Qualities decrease by row
        quality=copy(reshape(collect(1800:-1:1), 60, 30)')
    )

    @testset "Grid fields" begin
        @test size(g) == (30, 60)
        @test g.transitionlikelihood[1000:1002, 1000:1002] == [
            0.0 0.5 0.0
            0.5 0.0 0.5
            0.0 0.5 0.0]
        @test ConScape.sourceids(g)[1000:1002] == [
            CartesianIndex(10, 34),
            CartesianIndex(11, 34),
            CartesianIndex(12, 34)]
        @test ConScape.sourcequality(g)[20:22, 30:32] == [
              0.0   0.0   0.0
            571.0 570.0 569.0
            511.0 510.0 509.0]
        @test g.targetquality[20:22, 30:32] == [
              0.0   0.0   0.0
            571.0 570.0 569.0
            511.0 510.0 509.0]
    end

    problem = ConScape.Problem(
        measures = (
            kld = KullbackLeiblerDivergence(),
            betq = Betweenness(QualityWeighted()),
            betk = Betweenness(QualityAndProximityWeighted()),
            ebetq = EdgeBetweenness(QualityWeighted()),
            ebetk = EdgeBetweenness(QualityAndProximityWeighted()),
            ch = FunctionalHabitat(),
            pmp = PowerMeanProximity(),
            sp = SurvivalProbability(),
            fed = FreeEnergyDistance(),
            ex = ExpectedCost(),
        ),
        movement=RandomisedShortestPath(ExpectedCost(); theta=θ, diagvalue=0.0)
    )

    targetinit = init(init(problem, g), 1, 100)

    @testset "init fields" begin
        @test ConScape.transitioncost(targetinit).nzval[end-2:end] ≈ [
            1.039720770839918
            0.6931471805599453
            0.6931471805599453]
        @test targetinit.P.nzval[end-2:end] ≈ [
            0.10355339059327376,
            0.22654091966098644,
            0.22654091966098644]
        @test targetinit.W.nzval[end-2:end] ≈ [
            0.08411148966019986,
            0.19721532522049376,
            0.19721532522049376]
    end

    results = solve(problem, g)
    @testset "Test mean_kl_divergence" begin
        @test results.kld[] ≈ 2.4405084252728125e13
    end

    @testset "Test betweenness" begin
        # Check that summed edge betweennesses corresponds to node betweennesses:
        bet_node = results.betq
        bet_edge = results.ebetq
        bet = fill(NaN, size(g))
        bet[ConScape.sourceids(g)] = sum(results.ebetq; dims=2)
        # Edge betweenness is broken
        @test_broken bet ≈ bet_node
    end
    @testset "FunctionalHabitat" begin
        kw = (; theta=0.2, diagvalue=0.0)
        @test solve(FunctionalHabitat(), RSP(ExpectedCost(); distance_transformation=ExpMinus(), kw...), g)[28:30, 58:60]' ≈ [
             11082.654882969266 2664.916100189486 89.420910249988
             10340.977912804196 2465.918728844169 56.970111157896
             11119.132467660969 2662.969749775032 33.280379014217]

        @test solve(FunctionalHabitat(), RSP(FreeEnergyDistance(); distance_transformation=ExpMinus(), kw...), g)[28:30,58:60] ≈ [
                 93.0825   140.907    362.669
                 41.1656    63.2089   159.685
                 3.65643    4.04458    4.23555] rtol=1e-3

        # TODO some defaults must have been differen here... 
        @test solve(FunctionalHabitat(), init(RSP(SurvivalProbability(); kw...), g))[28:30,58:60] ≈ [
                 74141.1   72293.9    72294.7
                 27854.8   27066.5    26995.2
                 1151.38    765.195    391.131] rtol=1e-3

        @test solve(FunctionalHabitat(), init(RSP(PowerMeanProximity(); kw...), g))[28:30,58:60] ≈ [
                 93.0825   140.907    362.669
                 41.1656    63.2089   159.685
                 3.65643    4.04458    4.23555] rtol=1e-3
    end

    @testset "mean_lc_kl_divergence" begin
        @test_broken solve(KullbackLeiblerDivergence(), LCP(), g)[] ≈ 1.0667623231698838e14
    end

    # Eigmax doesn't work per-target
    # @testset "eigmax, proximity_measure=$proximity_measure" for
    #     (proximity_measure, val) in ((ExpectedCost()       , 5.576850282179157e6),
    #                                     (FreeEnergyDistance(), 3.2799955467465096e6),
    #                                     (SurvivalProbability(), 1.3475609129305437e7),
    #                                     (PowerMeanProximity(), 3.279995546746518e6))
    #     proximity_measure = ExpectedCost()
    #     rsp = RSP(proximity_measure; theta=θ)
    #     vˡ, λ, vʳ = solve(EigMax(), rsp, g)

    #     # Compute the weighted proximity matrix to check results
    #     S = solve(proximity_measure, rsp, g)
    #     if connectivity_function <: DistanceFunction
    #         map!(ExpMinus(), S, S)
    #     end
    #     qSq = g.source_qualities[:] .* S .* grsp.g.target_qualities[:]'

    #     @test λ ≈ val
    #     @test qSq*vʳ ≈ vʳ*λ
    # end

    @testset "Coarse graining: merging pixels to landmarks" begin
        g_coarse = ConScape.permeable_wall_sim(30, 60; corridorwidths=(3,2),
            # Qualities decrease by row
            quality=copy(reshape(collect(1800:-1:1), 60, 30)'), grain=3
        )

        @test ConScape.targetquality(g_coarse)[1:5, 1:5] ≈ [
            0.0     0.0 0.0 0.0     0.0
            0.0 15651.0 0.0 0.0 15624.0
            0.0     0.0 0.0 0.0     0.0
            0.0     0.0 0.0 0.0     0.0
            0.0 14031.0 0.0 0.0 14004.0]

        # @testset "eigmax, proximity_measure=$proximity_measure" for
        #     (proximity_measure, val) in ((ExpectedCost(), 2.7249231390873615e7),
        #                                     (FreeEnergyDistance(), 2.7217089009360086e7),
        #                                     (SurvivalProbability(), 3.0731253357215535e7),
        #                                     (PowerMeanProximity(), 2.7217089009360246e7))

        #     vˡ, λ, vʳ = solve(EigMax(), init(RSP(proximity_measure; theta=θ), g_coarse_rsp))
        #     @test λ ≈ val
        # end

        @testset "FunctionalHabitat" begin
            @testset "expected_cost" begin
                rsp = RSP(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ)
                fh_rsp = solve(FunctionalHabitat(), rsp, g_coarse)
                fh_g = solve(
                    FunctionalHabitat(), 
                    RSP(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ),
                    g_coarse,
                )
                fh_g_approx = solve(
                    FunctionalHabitat(),
                    RSP(ExpectedCost(); distance_transformation=ExpMinus(), theta=θ, approx=true),
                    g_coarse,
                )

                @test fh_g ≈ fh_rsp
                @test fh_g ≈ fh_g_approx rtol=0.8 # Very rough approximation
            end

            @testset "least_cost_distance" begin
                lc = LeastCostPath(; distance_transformation=ExpMinus())
                fh_rsp_lc = solve(FunctionalHabitat(), lc, g_coarse)
                fh_g_lc = solve(FunctionalHabitat(), lc, g_coarse)

                @test fh_g_lc ≈ fh_rsp_lc
            end
        end
    end

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))
    end
end

@testset "wall_landmark1" begin
    sq = copy(reshape(collect(1800:-1:1), 60, 30)')
    g = ConScape.permeable_wall_sim(30, 60;
        corridorwidths=(3,2),
        sourcequality=sq,
        targetquality=sparse(
            [10, 20, 10, 20],
            [15, 15, 45, 45],
            [sq[10, 15], sq[20, 15], sq[10, 45], sq[20, 45]],
            30, 60)
    )

    g2 = ConScape.permeable_wall_sim(30, 60,
        corridorwidths=(3,2),
        quality=sq
    )

    rsp = RSP(; theta=0.2)

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))
    end

    @testset "Landmark approach" begin
        @test solve(Betweenness(QualityWeighted()), rsp, g)[9:11, 30:32] ≈
            [1.35257193796979e9 1.3112254944853191e9 1.3525448385844798e9
             1.7383632661402326e9 1.9571251417867596e9 1.7385247019409044e9
             1.352382919812123e9 1.3103077614483771e9 1.3520848636655023e9]

    end
end

@testset "wall_landmark2" begin
    sq = copy(reshape(collect(1800:-1:1), 60, 30)')
    tmpgrid = [CartesianIndex((i,j)) for i in 1:2:30, j in 1:2:60]
    landmarks = sparse(
        [i[1] for i in tmpgrid][:],
        [i[2] for i in tmpgrid][:],
        [sq[i] for i in tmpgrid][:], 30, 60)

    g = ConScape.permeable_wall_sim(30, 60;
        corridorwidths=(3,2),
        sourcequality=sq,
        targetquality=landmarks
    )
    rsp = RSP(; theta=0.2, distance_transformation=ExpMinus())

    @testset "Landmark approach" begin
        @test solve(Betweenness(QualityAndProximityWeighted()), rsp, g)[9:11, 30:32] ≈
            [1.6153674943888483e6 693690.2564610258    1.6097137526944755e6
             1.8168095466336345e6 1.8166090537379407e6 1.8108940319968446e6
             1.41753770380708e6   668884.5700736387    1.412290291817482e6 ]
    end
end
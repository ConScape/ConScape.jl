@testset "wall full" begin
    θ = 0.2

    # Create the same landscape in Julia
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
    # Qualities decrease by row
        qualities=copy(reshape(collect(1800:-1:1), 60, 30)')
    )

    @testset "Grid fields" begin
        @test g.ncols == 60
        @test g.nrows == 30
        @test g.affinities[1000:1002, 1000:1002] == [
            0.0 0.5 0.0
            0.5 0.0 0.5
            0.0 0.5 0.0]
        @test g.id_to_grid_coordinate_list[1000:1002] == [
            CartesianIndex(10, 34),
            CartesianIndex(11, 34),
            CartesianIndex(12, 34)]
        @test g.source_qualities[20:22, 30:32] == [
              0.0   0.0   0.0
            571.0 570.0 569.0
            511.0 510.0 509.0]
        @test g.target_qualities[20:22, 30:32] == [
              0.0   0.0   0.0
            571.0 570.0 569.0
            511.0 510.0 509.0]
    end

    grsp = ConScape.GridRSP(g, θ=θ)

    @testset "GridRSP fields" begin
        @test grsp.g.costmatrix.nzval[end-2:end] ≈ [
            1.039720770839918
            0.6931471805599453
            0.6931471805599453]
        @test grsp.Pref.nzval[end-2:end] ≈ [
            0.10355339059327376,
            0.22654091966098644,
            0.22654091966098644]
        @test grsp.W.nzval[end-2:end] ≈ [
            0.08411148966019986,
            0.19721532522049376,
            0.19721532522049376]
        @test grsp.Z[100:102,100:102] ≈ [
            1.229380788700237   0.29706639745977187 0.11556093957432793
            0.29706639745977187 1.22938026597041    0.297066141383724
            0.11556093957432793 0.29706614138372406 1.2293801404819298]
    end

    @testset "Test mean_kl_divergence" begin
        @test_broken ConScape.mean_kl_divergence(grsp) ≈ 2.4405084252728125e13
    end

    @testset "Test betweenness" begin
        # Check that summed edge betweennesses corresponds to node betweennesses:
        bet_node = ConScape.betweenness_qweighted(grsp)
        bet_edge = ConScape.edge_betweenness_qweighted(grsp)
        bet = fill(NaN, grsp.g.nrows, grsp.g.ncols)
        for (i, v) in enumerate(sum(bet_edge,dims=2))
            bet[grsp.g.id_to_grid_coordinate_list[i]] = v
        end

        @test bet ≈ bet_node
    end

    @testset "connected_habitat" begin
        @test ConScape.ConScape.connected_habitat(grsp, diagvalue=0.0)[28:30,58:60]' ≈
            [11082.654882969266 2664.916100189486 89.420910249988
             10340.977912804196 2465.918728844169 56.970111157896
             11119.132467660969 2662.969749775032 33.280379014217]

       @test ConScape.ConScape.connected_habitat(grsp, diagvalue=0.0,
            connectivity_function=ConScape.free_energy_distance)[28:30,58:60] ≈ [
                 93.0825   140.907    362.669
                 41.1656    63.2089   159.685
                  3.65643    4.04458    4.23555] rtol=1e-3

       @test ConScape.ConScape.connected_habitat(grsp, diagvalue=0.0,
            connectivity_function=ConScape.survival_probability)[28:30,58:60] ≈ [
                 74141.1   72293.9    72294.7
                 27854.8   27066.5    26995.2
                  1151.38    765.195    391.131] rtol=1e-3

        @test ConScape.ConScape.connected_habitat(grsp, diagvalue=0.0,
            connectivity_function=ConScape.power_mean_proximity)[28:30,58:60] ≈ [
                 93.0825   140.907    362.669
                 41.1656    63.2089   159.685
                  3.65643    4.04458    4.23555] rtol=1e-3
    end

    @testset "mean_lc_kl_divergence" begin
        @test ConScape.ConScape.mean_lc_kl_divergence(grsp) ≈ 1.0667623231698838e14
    end

    @testset "eigmax, connectivity_function=$connectivity_function" for
        (connectivity_function, val) in ((ConScape.expected_cost       , 5.576850282179157e6),
                                         (ConScape.free_energy_distance, 3.2799955467465096e6),
                                         (ConScape.survival_probability, 1.3475609129305437e7),
                                         (ConScape.power_mean_proximity, 3.279995546746518e6))

        vˡ, λ, vʳ = ConScape.eigmax(grsp; connectivity_function)

        # Compute the weighted proximity matrix to check results
        S   = connectivity_function(grsp)
        if connectivity_function <: ConScape.DistanceFunction
            map!(ConScape.ExpMinus(), S, S)
        end
        qSq = grsp.g.source_qualities[:] .* S .* grsp.g.target_qualities[:]'

        @test λ ≈ val
        @test qSq*vʳ ≈ vʳ*λ
    end

    @testset "Coarse graining: merging pixels to landmarks" begin
        g_coarse = ConScape.Grid(
            size(g)...,
            affinities=g.affinities,
            source_qualities=g.source_qualities,
            target_qualities=ConScape.coarse_graining(g, 3))

        @test g_coarse.target_qualities[1:5, 1:5] ≈ [
            0.0     0.0 0.0 0.0     0.0
            0.0 15651.0 0.0 0.0 15624.0
            0.0     0.0 0.0 0.0     0.0
            0.0     0.0 0.0 0.0     0.0
            0.0 14031.0 0.0 0.0 14004.0]

        g_coarse_rsp = ConScape.GridRSP(g_coarse, θ=θ)

        @testset "eigmax, connectivity_function=$connectivity_function" for
            (connectivity_function, val) in ((ConScape.expected_cost       , 2.7249231390873615e7),
                                             (ConScape.free_energy_distance, 2.7217089009360086e7),
                                             (ConScape.survival_probability, 3.0731253357215535e7),
                                             (ConScape.power_mean_proximity, 2.7217089009360246e7))

            vˡ, λ, vʳ = ConScape.eigmax(g_coarse_rsp,
                connectivity_function=connectivity_function)
            @test λ ≈ val
        end

        @testset "connected_habitat" begin
            @testset "expected_cost" begin
                ch_rsp = ConScape.connected_habitat(g_coarse_rsp)
                ch_g = ConScape.connected_habitat(
                    g_coarse;
                    distance_transformation=ConScape.ExpMinus(),
                    θ=θ)
                ch_g_approx = ConScape.connected_habitat(
                    g_coarse;
                    distance_transformation=ConScape.ExpMinus(),
                    θ=θ,
                    approx=true)

                @test ch_g ≈ ch_rsp
                @test ch_g ≈ ch_g_approx rtol=0.8 # Very rough approximation
            end

            @testset "least_cost_distance" begin
                ch_rsp_lc = ConScape.connected_habitat(
                    g_coarse_rsp;
                    connectivity_function=ConScape.least_cost_distance)
                ch_g_lc = ConScape.connected_habitat(
                    g_coarse;
                    connectivity_function=ConScape.least_cost_distance,
                    distance_transformation=ConScape.ExpMinus())

                @test ch_g_lc ≈ ch_rsp_lc
            end
        end
    end

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))
    end
end

@testset "wall_landmark1" begin
    sq = copy(reshape(collect(1800:-1:1), 60, 30)')

    g = ConScape.perm_wall_sim(
        30,
        60,
        corridorwidths=(3,2),
        source_qualities=sq,
        target_qualities=sparse(
            [10, 20, 10, 20],
            [15, 15, 45, 45],
            [sq[10, 15], sq[20, 15], sq[10, 45], sq[20, 45]],
            30, 60))

    g2 = ConScape.perm_wall_sim(
        30,
        60,
        corridorwidths=(3,2),
        qualities=sq)

    grsp = ConScape.GridRSP(g, θ=0.2)

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))
    end

    @testset "Landmark approach" begin
        @test ConScape.betweenness_qweighted(grsp)[9:11, 30:32] ≈
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

    g = ConScape.perm_wall_sim(
        30,
        60,
        corridorwidths=(3,2),
        source_qualities=sq,
        target_qualities=landmarks)

    grsp = ConScape.GridRSP(g, θ=0.2)

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))
    end

    @testset "Landmark approach" begin
        @test ConScape.betweenness_kweighted(grsp)[9:11, 30:32] ≈
            [1.6153674943888483e6 693690.2564610258    1.6097137526944755e6
             1.8168095466336345e6 1.8166090537379407e6 1.8108940319968446e6
             1.41753770380708e6   668884.5700736387    1.412290291817482e6 ]
    end
end


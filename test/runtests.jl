using ConScape, Test, SparseArrays

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

@testset "sno_2000" begin
    landscape = "sno_2000"
    θ = 0.1

    affinity_raster, _ = ConScape.readasc(joinpath(datadir, "affinities_$landscape.asc"))

    @testset "writeasc and readasc roundtrip" for
        (xllcorner, yllcorner, cellsize) in ((1, 2, 7),
                                             (1.1, 2.2, 7.7))

        filename = joinpath(_tempdir, "sno.asc")
        ConScape.writeasc(filename, affinity_raster,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize)

        _raster, _meta = ConScape.readasc(filename)

        @test isequal(_raster, affinity_raster) # contain NaNs
        @test _meta["xllcorner"] == xllcorner
        @test _meta["yllcorner"] == yllcorner
        @test _meta["cellsize"] == cellsize

        filename2 = joinpath(_tempdir, "sno2.asc")

        ConScape.writeasc(filename2, affinity_raster, _meta)

        _raster2, _meta2 = ConScape.readasc(filename2)
        @test isequal(_raster, _raster2) # contain NaNs
        @test all(_meta[k] == _meta2[k] for k in keys(_meta))
    end

    affinities = ConScape.graph_matrix_from_raster(affinity_raster)
    @test affinities[1000:1002, 1000:1002] == [
        0.0               0.00031508895477488 0.0
        0.133336775193571 0.0                 0.00119533310704962
        0.0               0.00031508895477488 0.0]

    qualities , _ = ConScape.readasc(joinpath(datadir, "qualities_$landscape.asc"))
    # FIXME! We'd have to handle this somehow in the library
    @test_broken isnan.(affinity_raster) == isnan.(qualities)
    qualities[(affinity_raster .> 0) .& isnan.(qualities)] .= 1e-20

    g = ConScape.Grid(size(affinity_raster)...,
        affinities=affinities,
        qualities=qualities
        )

    @testset "Grid fields" begin
        @test g.ncols == 59
        @test g.nrows == 44
        @test g.affinities[1000:1002, 1000:1002] == [
            0.0               0.557963581550866 0.0
            0.607917269319296 0.0               0.273319838512689
            0.0               0.557963581550866 0.0]
        @test g.id_to_grid_coordinate_list[1000:1002] == [
            CartesianIndex(37, 41), CartesianIndex(38, 41), CartesianIndex(39, 41)]
        @test g.source_qualities[30:32, 30:32] == [
            0.151232712594792 0.146546460358077  0.00748122241586316
            0.170905506269773 0.0532220626743219 0.00309705330379074
            0.10284506027383  0.059244283127482  0.0361260830667015]
        @test g.target_qualities[30:32, 30:32] == [
            0.151232712594792 0.146546460358077  0.00748122241586316
            0.170905506269773 0.0532220626743219 0.00309705330379074
            0.10284506027383  0.059244283127482  0.0361260830667015]
        @test g.costfunction == ConScape.MinusLog()
        @test g.costmatrix.nzval[end-2:end] ≈ [
            17.339919976251554
            16.99334638597158
            17.339919976251554]
    end

    @testset "Grid plotting" begin
        @test ConScape.plot_indegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_outdegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_values(g,ones(length(g.id_to_grid_coordinate_list))) isa ConScape.Plots.Plot
    end

    grsp = ConScape.GridRSP(g, θ=θ)

    @testset "GridRSP fields" begin
        @test grsp.Pref.nzval[end-2:end] ≈ [
            0.00031266870414554466,
            0.00018055948867712768,
            0.0001228960184654185]
        @test grsp.W.nzval[end-2:end] ≈ [
            5.521044625610329e-5,
            3.300719810371289e-5,
            2.1700745653826738e-5]
        @test grsp.Z[100:102,111:113] ≈ [
            0.00016367398568399748 0.00015329797258788392 0.00023129137088760695
            6.432239855543766e-5   5.4944307498337036e-5  7.336349355801132e-5
            2.421703672476501e-5   1.953276891351377e-5   2.4440692415300965e-5]
    end

    @testset "Test mean_kl_divergence" begin
        @test ConScape.mean_kl_divergence(grsp) ≈ 323895.3828183995
    end

    @testset "test adjacency creation with $nn neighbors, $w weighting and $mt" for
        nn in (ConScape.N4, ConScape.N8),
            w in (ConScape.TargetWeight, ConScape.AverageWeight),
                mt in (ConScape.AffinityMatrix, ConScape.CostMatrix)
# No need to test this on sno_100 and doesn't deepend on θ
# FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
                    @test ConScape.graph_matrix_from_raster(
                        affinity_raster,
                        neighbors=nn,
                        weight=w,
                        matrix_type=mt) isa ConScape.SparseMatrixCSC
    end

    @testset "Test betweenness" begin
        @testset "q-weighted" begin
            bet = ConScape.betweenness_qweighted(grsp)
            @test bet[21:23, 21:23] ≈ [
                1930.1334372152335  256.91061166392745 2866.2998374065373
                4911.996715311025  1835.991238248377    720.755518530375
                4641.815380725279  3365.3296878569213   477.1085971945757]
        end

        @testset "k-weighted" begin
            bet = ConScape.betweenness_kweighted(grsp, diagvalue=1.)
            @test bet[21:23, 31:33] ≈ [
                0.04063917813171917 0.06843246983487516 0.08862506281612659
                0.03684621201600996 0.10352876485995872 0.1255652231824746
                0.03190640567704462 0.13832814750469344 0.1961393152256104]

            # Check that summed edge betweennesses corresponds to node betweennesses:
            bet_edge = ConScape.edge_betweenness_kweighted(grsp, diagvalue=1.)
            bet_edge_sum = fill(NaN, grsp.g.nrows, grsp.g.ncols)
            for (i, v) in enumerate(sum(bet_edge,dims=2))
                bet_edge_sum[grsp.g.id_to_grid_coordinate_list[i]] = v
            end
            @test bet_edge_sum[21:23, 31:33] ≈ bet[21:23, 31:33]

            # This is a regression test based on values that we currently believe to be correct
            bet = ConScape.betweenness_kweighted(grsp, distance_transformation=t -> exp(-t/50))
            @test bet[21:23, 31:33] ≈ [
                980.5828087688377 1307.981162399926 1602.8445739784497
                826.0710054834001 1883.0940077789735 1935.4450344630702
                676.9212075214159 2228.2700913772774 2884.0409495023364]

            @test ConScape.betweenness_kweighted(grsp, distance_transformation=one)[g.id_to_grid_coordinate_list] ≈
                ConScape.betweenness_qweighted(grsp)[g.id_to_grid_coordinate_list]

            @test ConScape.edge_betweenness_kweighted(grsp, distance_transformation=one) ≈
                ConScape.edge_betweenness_qweighted(grsp)
        end
    end

    @testset "connected_habitat" begin
        ch = ConScape.connected_habitat(grsp)
        @test ch isa Matrix{Float64}
        @test size(ch) == size(grsp.g.source_qualities)

        cl = ConScape.connected_habitat(grsp, CartesianIndex((20,20)))
        @test cl isa Matrix{Float64}
        @test sum(replace(cl, NaN => 0.0)) ≈ 109.4795495188798
    end

    @testset "mean_lc_kl_divergence" begin
        @test ConScape.ConScape.mean_lc_kl_divergence(grsp) ≈ 1.5660600315073947e6
    end

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", grsp)
        @test occursin("GridRSP", String(take!(b)))
    end
end


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

    @testset "Grid plotting" begin
        @test ConScape.plot_indegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_outdegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_values(g,ones(length(g.id_to_grid_coordinate_list))) isa ConScape.Plots.Plot
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
        @test ConScape.mean_kl_divergence(grsp) ≈ 2.4405084252728125e13
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

        vˡ, λ, vʳ = ConScape.eigmax(grsp,
            connectivity_function=connectivity_function)

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

        b = IOBuffer()
        show(b, "text/html", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", grsp)
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

    @testset "Grid plotting" begin
        @test ConScape.plot_indegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_outdegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_values(g,ones(length(g.id_to_grid_coordinate_list))) isa ConScape.Plots.Plot
    end

    grsp = ConScape.GridRSP(g, θ=0.2)

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", grsp)
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

    @testset "Grid plotting" begin
        @test ConScape.plot_indegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_outdegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_values(g,ones(length(g.id_to_grid_coordinate_list))) isa ConScape.Plots.Plot
    end

    grsp = ConScape.GridRSP(g, θ=0.2)

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", grsp)
        @test occursin("GridRSP", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", grsp)
        @test occursin("GridRSP", String(take!(b)))
    end

    @testset "Landmark approach" begin
        @test ConScape.betweenness_kweighted(grsp)[9:11, 30:32] ≈
            [1.6153674943888483e6 693690.2564610258    1.6097137526944755e6
             1.8168095466336345e6 1.8166090537379407e6 1.8108940319968446e6
             1.41753770380708e6   668884.5700736387    1.412290291817482e6 ]
    end
end


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

    g1 = ConScape.Grid(size(l1)..., affinities=ConScape.graph_matrix_from_raster(l1), prune=false)
    g2 = ConScape.Grid(size(l2)..., affinities=ConScape.graph_matrix_from_raster(l2))

    @test !ConScape.is_strongly_connected(g1)
    @test ConScape.is_strongly_connected(g2)
    for f in fieldnames(typeof(g1))
        @test getfield(ConScape.largest_subgraph(g1), f) == getfield(g2, f)
    end
end

@testset "least cost distance" begin
    r = [1/4 0 1/2 1/4
         1/4 0 1/2 1/4
         1/4 0 1/2 1/4
         1/4 0 1/2 1/4]

    a = ConScape.graph_matrix_from_raster(r, neighbors=ConScape.N4)
    c = copy(a)
    c.nzval .= 1/2

    @testset "_cost: $_cost, op: $op, prune: $prune" for
        (_cost, op) in ((ConScape.MinusLog(), <), (c, ==)),
            prune in (true, false)

        g = ConScape.Grid(size(r)..., affinities=a, costs=_cost, prune=prune)
        lc = ConScape.least_cost_distance(g)
        @test prune || all(isinf, lc[1:8, 9:16])
        # since (4, 3) -> (4, 4) has higher affinity than (3, 4) -> (4, 4), i.e. lower cost
        # when costs=MinusLog() and identical affinities and costs when using the cost matrix c
        if prune
            # pruned landscape has size (4, 2)
            @test op(lc[(1 - 1)*4 + 4, 8], lc[(2 - 1)*4 + 3, 8])
        else
            # full landscape has size (4, 4)
            @test op(lc[(3 - 1)*4 + 4, 16], lc[(4 - 1)*4 + 3, 16])
        end
    end
end

@testset "Distances and proximities" begin
    l = [1 1
         1 1]

    a = ConScape.graph_matrix_from_raster(l, neighbors=ConScape.N4)

    c = ConScape.graph_matrix_from_raster(
        l,
        neighbors=ConScape.N4,
        matrix_type=ConScape.CostMatrix)

    @testset "check shapes of affinity and cost matrices" begin
        @test_throws ArgumentError("grid size (2, 2) is incompatible with size of affinity matrix (3, 3)") ConScape.Grid(
            size(l)...,
            affinities=a[1:end-1, 1:end-1],
            costs=c)

        @test_throws ArgumentError("grid size (2, 2) is incompatible with size of cost matrix (3, 3)") ConScape.Grid(
            size(l)...,
            affinities=a,
            costs=c[1:end-1, 1:end-1])
    end

    g = ConScape.Grid(
        size(l)...,
        affinities=a,
        costs=c)

    grsp = ConScape.GridRSP(g, θ=2.)

    free_energy_grsp = ConScape.free_energy_distance(grsp)
    @test free_energy_grsp ≈ [
      0.0       1.34197   1.34197   2.34197
      1.34197   0.0       2.34197   1.34197
      1.34197   2.34197   0.0       1.34197
      2.34197   1.34197   1.34197   0.0     ] atol=1e-4
    @test ConScape.free_energy_distance(g; θ=grsp.θ) ≈ free_energy_grsp

    excepted_cost_grsp = ConScape.expected_cost(grsp)
    @test excepted_cost_grsp ≈ [
      0.0      1.01848  1.01848  2.01848
      1.01848  0.0      2.01848  1.01848
      1.01848  2.01848  0.0      1.01848
      2.01848  1.01848  1.01848  0.0 ] atol=1e-4
    @test ConScape.expected_cost(g; θ=grsp.θ) ≈ excepted_cost_grsp

    survival_probability_grsp = ConScape.survival_probability(grsp)
    @test survival_probability_grsp ≈ [
      1.0         0.0682931   0.0682931   0.00924246
      0.0682931   1.0         0.00924246  0.0682931
      0.0682931   0.00924246  1.0         0.0682931
      0.00924246  0.0682931   0.0682931   1.0    ] atol=1e-4
    @test ConScape.survival_probability(g; θ=grsp.θ) ≈ survival_probability_grsp

    power_mean_proximity_grsp = ConScape.power_mean_proximity(grsp)
    @test power_mean_proximity_grsp ≈ [
      1.0        0.261329   0.261329   0.0961377
      0.261329   1.0        0.0961377  0.261329
      0.261329   0.0961377  1.0        0.261329
      0.0961377  0.261329   0.261329   1.0      ] atol=1e-4
    @test ConScape.power_mean_proximity(g; θ=grsp.θ) ≈ power_mean_proximity_grsp
end


@testset "custom scaling function in k-weighted betweenness" begin
    l = rand(4, 4)
    q = rand(4, 4)

    g = ConScape.Grid(size(l)..., affinities=ConScape.graph_matrix_from_raster(l))
    grsp = ConScape.GridRSP(g, θ=0.2)

    @test ConScape.betweenness_kweighted(grsp) == ConScape.betweenness_kweighted(grsp; distance_transformation=t -> exp(-t))
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

    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2))
    grsp = ConScape.GridRSP(g, θ=0.2)
    @test ConScape.least_cost_kl_divergence(grsp, (25,50))[10,10] ≈ 80.63375074079197
end

# FIXME! Computation is currently very slow so we have to use a reduced landscape
@testset "Criticality" begin
    m, n = 10, 15
    g = ConScape.perm_wall_sim(m, n, corridorwidths=(2,2),
        # Qualities decrease by row
        qualities=copy(reshape(collect(m*n:-1:1), n, m)'))
    grsp = ConScape.GridRSP(g, θ=0.2)
    @test sum(t -> isnan(t) ? 0.0 : t, ConScape.criticality(grsp) .< -1e-5) == 0
end

@testset "pass cost matrix instead of function" begin
    m, n = 10, 15

    _g = ConScape.perm_wall_sim(m, n, corridorwidths=(2,2),
        # Qualities decrease by row
        qualities=copy(reshape(collect(m*n:-1:1), n, m)'))

    g = ConScape.Grid(m, n,
        affinities=_g.affinities,
        qualities=_g.source_qualities,
        costs=ConScape.MinusLog())
    grsp = ConScape.GridRSP(g, θ=0.2)

    g_with_costs = ConScape.Grid(m, n,
        affinities=_g.affinities,
        qualities=_g.source_qualities,
        costs=ConScape.mapnz(ConScape.MinusLog(), _g.affinities))
    grsp_with_costs = ConScape.GridRSP(g_with_costs, θ=0.2)

    @test g_with_costs.costfunction === nothing

    @test ConScape.betweenness_qweighted(grsp) == ConScape.betweenness_qweighted(grsp_with_costs)

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
    affinities = ConScape.graph_matrix_from_raster(l)

    for c in [ConScape.MinusLog(),
              ConScape.ExpMinus(),
              ConScape.Inv(),
              ConScape.OddsAgainst(),
              ConScape.OddsFor()]

        g = ConScape.Grid(
            size(l)...,
            affinities=affinities,
            costs=c)

        h_c = ConScape.GridRSP(g, θ=0.2)
        @test h_c isa ConScape.GridRSP
    end

    affinities[1,2] = 1.1 # Causes negative cost for C[1,2] when costs=MinusLog
    @test_throws ArgumentError ConScape.Grid(
        size(l)...,
        affinities=affinities,
        costs=ConScape.MinusLog()) # should raise error, as C[1,2]<0
end

@testset "Avoid NaNs when Z has tiny values" begin
    mov_prob, meta_p = ConScape.readasc(joinpath(datadir, "mov_prob_1000.asc"))

    q = zeros(size(mov_prob))
    q[60,70]   = 1
    q[50, 105] = 1
    g = ConScape.Grid(size(mov_prob)...,
        affinities=ConScape.graph_matrix_from_raster(mov_prob),
        qualities=q,
        costs=ConScape.MinusLog());
    grsp = ConScape.GridRSP(g, θ=2.5);
    betw = ConScape.betweenness_qweighted(grsp)
    @test betw[58:60, 78:80] ≈ [
        0.397426   0.170278   0.348822
        1.42686    1.65378    1.419
        0.0554379  0.0192699  0.185261] rtol=1e-3
end

@testset "Avoid overflow in k-weighted betweenness" begin
    mov_prob, meta_p = ConScape.readasc(joinpath(datadir, "mov_prob_200.asc"))
    hab_qual, meta_q = ConScape.readasc(joinpath(datadir, "hab_qual_200.asc"))

    # FIXME! We'd have to handle this somehow in the library
    @test_broken isnan.(mov_prob) == isnan.(hab_qual)
    non_matches = findall(xor.(isnan.(mov_prob), isnan.(hab_qual)))
    mov_prob[non_matches] .= 1e-20
    hab_qual[non_matches] .= 1e-20

    g = ConScape.Grid(size(mov_prob)...,
        affinities=ConScape.graph_matrix_from_raster(mov_prob),
        qualities=hab_qual,
        costs=ConScape.MinusLog())

    g_coarse = ConScape.Grid(size(mov_prob)...,
        affinities=ConScape.graph_matrix_from_raster(mov_prob),
        source_qualities=hab_qual,
        target_qualities=ConScape.coarse_graining(g, 200),
        costs=ConScape.MinusLog())

    h_coarse = ConScape.GridRSP(g_coarse, θ=1.0)

    kbetw = @time ConScape.betweenness_kweighted(h_coarse, distance_transformation=x -> exp(-x/100))
    @test count(!isnan, kbetw) == 128234
end

@testset "Test that cost edges are contained in the affinity edges" begin
    @test_throws ArgumentError("cost graph contains edges not present in the affinity graph") ConScape.Grid(2, 2,
        affinities=sparse(
            [3, 4, 1, 4, 2, 3],
            [1, 2, 3, 3, 4, 4],
            [1.0, 1, 1, 1, 1, 1]),
        costs=sparse(
            [2, 3, 1, 4, 1, 4, 2, 3],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1.0, 1, 1, 1, 1, 1, 1, 1]))
end

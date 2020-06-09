using ConScape, Test, SparseArrays

datadir = joinpath(@__DIR__(), "..", "data")

@testset "test landscape: $landscape" for
    # FIXME! Enable testing of sno_1000 landscape with landmarks. The full landscape is too large for CI
    landscape in ("wall_full", "wall_landmark1", "wall_landmark2", "sno_2000",#= "sno_1000"=#),
        β in (0.1, 0.2)

    if landscape == "wall_full"
        # Create the same landscape in Julia
        g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
                               # Qualities decrease by row
                               qualities=copy(reshape(collect(1800:-1:1), 60, 30)')
                               )
        h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=β)
    elseif landscape == "wall_landmark1"
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
        h = ConScape.Habitat(g, β=0.2)
    elseif landscape == "wall_landmark2"
        sq = copy(reshape(collect(1800:-1:1), 60, 30)')
        tmpgrid = [CartesianIndex((i,j)) for i in 1:2:30, j in 1:2:60]
        landmarks = sparse([i[1] for i in tmpgrid][:],
                           [i[2] for i in tmpgrid][:],
                           [sq[i] for i in tmpgrid][:], 30, 60)
        g = ConScape.perm_wall_sim(
            30,
            60,
            corridorwidths=(3,2),
            source_qualities=sq,
            target_qualities=landmarks)
        h = ConScape.Habitat(g, β=0.2)
    else
        affinities, _ = ConScape.readasc(joinpath(datadir, "affinities_$landscape.asc"))
        qualities , _ = ConScape.readasc(joinpath(datadir, "qualities_$landscape.asc"))
        g = ConScape.Grid(size(affinities)...,
                          landscape=ConScape.graph_matrix_from_raster(affinities),
                          qualities=qualities
                          )
        h = ConScape.Habitat(g, β=β)
    end

    @testset "Test mean_kl_divergence" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.mean_kl_divergence(h) ≈ 2.4405084252728125e13
        elseif landscape == "sno_2000" && β == 0.1
            @test ConScape.mean_kl_divergence(h) ≈ 323895.3828183995
        end
    end

    @testset "test adjacency creation with $nn neighbors, $w weighting and $mt" for
        nn in (ConScape.N4, ConScape.N8),
            w in (ConScape.TargetWeight, ConScape.AverageWeight),
                mt in (ConScape.AffinityMatrix, ConScape.CostMatrix)

        if landscape == "sno_2000" && β == 0.1 # No need to test this on sno_100 and doesn't deepend on β
            # FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
            @test ConScape.graph_matrix_from_raster(affinities, neighbors=nn, weight=w, matrix_type=mt) isa ConScape.SparseMatrixCSC
        end
    end

    @testset "Test betweenness" begin
        # FIXME Enable all combinations
        if landscape == "sno_2000" && β == 0.1
            # This is a regression test based on values that we currently believe to be correct
            bet = ConScape.RSP_betweenness_kweighted(h, diagvalue=1.)
            @test bet[21:23, 31:33] ≈ [0.04063917813171917 0.06843246983487516 0.08862506281612659
                                       0.03684621201600996 0.10352876485995872 0.1255652231824746
                                       0.03190640567704462 0.13832814750469344 0.1961393152256104]

            # Check that summed edge betweennesses corresponds to node betweennesses:
            bet_edge = ConScape.RSP_edge_betweenness_kweighted(h, diagvalue=1.)
            bet_edge_sum = fill(NaN, h.g.nrows, h.g.ncols)
            for (i, v) in enumerate(sum(bet_edge,dims=2))
                bet_edge_sum[h.g.id_to_grid_coordinate_list[i]] = v
            end
            @test bet_edge_sum[21:23, 31:33] ≈ bet[21:23, 31:33]

            # This is a regression test based on values that we currently believe to be correct
            bet = ConScape.RSP_betweenness_kweighted(h, invcost=t -> exp(-t/50))
            @test bet[21:23, 31:33] ≈ [980.5828087688377 1307.981162399926 1602.8445739784497
                                       826.0710054834001 1883.0940077789735 1935.4450344630702
                                       676.9212075214159 2228.2700913772774 2884.0409495023364]

            @test ConScape.RSP_betweenness_kweighted(h, invcost=one)[g.id_to_grid_coordinate_list] ≈
                    ConScape.RSP_betweenness_qweighted(h)[g.id_to_grid_coordinate_list]

            @test ConScape.RSP_edge_betweenness_kweighted(h, invcost=one) ≈
                    ConScape.RSP_edge_betweenness_qweighted(h)

        elseif landscape == "wall_full"
            # Check that summed edge betweennesses corresponds to node betweennesses:
            bet_node = ConScape.RSP_betweenness_qweighted(h)
            bet_edge = ConScape.RSP_edge_betweenness_qweighted(h)
            bet = fill(NaN, h.g.nrows, h.g.ncols)
            for (i, v) in enumerate(sum(bet_edge,dims=2))
                bet[h.g.id_to_grid_coordinate_list[i]] = v
            end

            @test bet ≈ bet_node
        end
    end

    @testset "RSP_functionality" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.ConScape.RSP_functionality(h, diagvalue=0.0)[28:30,58:60]' ≈
                [11082.654882969266 2664.916100189486 89.420910249988
                 10340.977912804196 2465.918728844169 56.970111157896
                 11119.132467660969 2662.969749775032 33.280379014217]
        elseif landscape == "sno_2000" && β == 0.1
            hf = ConScape.RSP_functionality(h)
            @test hf isa SparseMatrixCSC
            @test size(hf) == size(h.g.source_qualities)
        end
    end

    @testset "mean_lc_kl_divergence" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.ConScape.mean_lc_kl_divergence(h) ≈ 1.0667623231698838e14
        elseif landscape == "sno_2000" && β == 0.1
            @test ConScape.ConScape.mean_lc_kl_divergence(h) ≈ 1.5660600315073947e6
        end
    end

    @testset "Grid plotting" begin
        @test ConScape.plot_indegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_outdegrees(g) isa ConScape.Plots.Plot
        @test ConScape.plot_values(g,ones(length(g.id_to_grid_coordinate_list))) isa ConScape.Plots.Plot
    end

    @testset "Show methods" begin
        b = IOBuffer()
        show(b, "text/plain", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/plain", h)
        @test occursin("Habitat", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", g)
        @test occursin("Grid", String(take!(b)))

        b = IOBuffer()
        show(b, "text/html", h)
        @test occursin("Habitat", String(take!(b)))
    end

    @testset "Landmark approach" begin

        if landscape == "wall_landmark1" && β == 0.2
            # Just a regression test but result looks visually correct
            @test ConScape.RSP_betweenness_qweighted(h)[9:11, 30:32] ≈
                    [1.35257193796979e9 1.3112254944853191e9 1.3525448385844798e9
                     1.7383632661402326e9 1.9571251417867596e9 1.7385247019409044e9
                     1.352382919812123e9 1.3103077614483771e9 1.3520848636655023e9]

        elseif landscape == "wall_landmark2" && β == 0.2
            @test ConScape.RSP_betweenness_kweighted(h)[9:11, 30:32] ≈
                    [1.6153674943888483e6 693690.2564610258    1.6097137526944755e6
                     1.8168095466336345e6 1.8166090537379407e6 1.8108940319968446e6
                     1.41753770380708e6   668884.5700736387    1.412290291817482e6 ]
        end
    end

    @testset "Coarse graining: merging pixels to landmarks" begin
        if landscape == "wall_full" && β == 0.1 # No need to test for all values of β
            g_coarse = ConScape.Grid(size(g)...,
                                     landscape=g.A,
                                     source_qualities=g.source_qualities,
                                     target_qualities=ConScape.coarse_graining(g, 3))

            @test g_coarse.target_qualities[1:5, 1:5] ≈ [0.0     0.0 0.0 0.0     0.0
                                                         0.0 15651.0 0.0 0.0 15624.0
                                                         0.0     0.0 0.0 0.0     0.0
                                                         0.0     0.0 0.0 0.0     0.0
                                                         0.0 14031.0 0.0 0.0 14004.0]
        end
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

    g1 = ConScape.Grid(size(l1)..., landscape=ConScape.graph_matrix_from_raster(l1))
    g2 = ConScape.Grid(size(l2)..., landscape=ConScape.graph_matrix_from_raster(l2))

    @test !ConScape.is_connected(g1)
    @test ConScape.is_connected(g2)
    for f in fieldnames(typeof(g1))
        @test getfield(ConScape.largest_subgraph(g1), f) == getfield(g2, f)
    end
end

@testset "least cost distance" begin
    l = [1/4 0 1/4 1/4
         1/4 0 1/4 1/4
         1/4 0 1/4 1/4
         1/4 0 1/4 1/4]

    g = ConScape.Grid(size(l)..., landscape=ConScape.graph_matrix_from_raster(l,neighbors=ConScape.N4))

    @test all(ConScape.least_cost_distance(g, (4,4)) .=== [Inf  NaN  1.0   0.75
                                                           Inf  NaN  0.75  0.5
                                                           Inf  NaN  0.5   0.25
                                                           Inf  NaN  0.25  0.0])
end

@testset "Other distances and proximities" begin
    l = [1 1
         1 1 ]

    g = ConScape.Grid(size(l)..., landscape=ConScape.graph_matrix_from_raster(l,neighbors=ConScape.N4))
    C = ConScape.graph_matrix_from_raster(l,neighbors=ConScape.N4,matrix_type=ConScape.CostMatrix)
    h = ConScape.Habitat(g, C=C, β=2.)

    @test maximum(abs.(ConScape.RSP_free_energy_distance(h) - [0.0       1.34197   1.34197   2.34197
                                                               1.34197   0.0       2.34197   1.34197
                                                               1.34197   2.34197   0.0       1.34197
                                                               2.34197   1.34197   1.34197   0.0     ])) < 1e-5

    @test maximum(abs.(ConScape.RSP_survival_probability(h) - [1.0         0.0682931   0.0682931   0.00924246
                                                               0.0682931   1.0         0.00924246  0.0682931
                                                               0.0682931   0.00924246  1.0         0.0682931
                                                               0.00924246  0.0682931   0.0682931   1.0    ])) < 1e-5

    @test maximum(abs.(ConScape.RSP_power_mean_proximity(h) - [1.0        0.261329   0.261329   0.0961377
                                                               0.261329   1.0        0.0961377  0.261329
                                                               0.261329   0.0961377  1.0        0.261329
                                                               0.0961377  0.261329   0.261329   1.0      ])) < 1e-5
end


@testset "custom scaling function in k-weighted betweenness" begin
    l = rand(4, 4)
    q = rand(4, 4)

    g = ConScape.Grid(size(l)..., landscape=ConScape.graph_matrix_from_raster(l))
    h = ConScape.Habitat(g, β=0.2)

    @test ConScape.RSP_betweenness_kweighted(h) == ConScape.RSP_betweenness_kweighted(h; invcost=t -> exp(-t))
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
    h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=0.2)
    @test ConScape.least_cost_kl_divergence(h, (25,50))[10,10] ≈ 80.63375074079197
end

# FIXME! Computation is currently very slow so we have to use a reduced landscape
@testset "Criticality" begin
    m, n = 10, 15
    g = ConScape.perm_wall_sim(m, n, corridorwidths=(2,2),
                               # Qualities decrease by row
                               qualities=copy(reshape(collect(m*n:-1:1), n, m)')
                               )
    h = ConScape.Habitat(g, β=0.2)
    @test sum(ConScape.RSP_criticality(h).nzval .< -1e-5) == 0
end

##
@testset "Sensitivity" begin
    m, n = 12,16
    # g = ConScape.perm_wall_sim(m, n, scaling=0.35, corridorwidths=(3,), corridorpositions=(0.55,))#,
                               # Qualities decrease by row
                               # qualities=copy(reshape(collect(m*n:-1:1), n, m)')
                               # )

    # g = ConScape.Grid(m, n, qualities=copy(reshape(collect(m*n:-1:1), n, m)'))
    g = ConScape.Grid(m, n) #, qualities=copy(reshape(collect(m*n:-1:1), n, m)'))
    g.A = 0.35 * g.A

    h = ConScape.Habitat(g, β=0.5)

    testnodes = [5, 20, 39, 67, 92]
    # testnodes = collect(1:5)
    S_comp = ConScape.LF_sensitivity(h)[1]
    S_testnodes = S_comp[[h.g.id_to_grid_coordinate_list[i] for i in testnodes]]

    S_simu = ConScape.LF_sensitivity_simulation(h,testnodes=testnodes)


    @test maximum(abs.(S_testnodes - S_simu)./abs.(S_testnodes)) < 1e-5


    S_comp_pm = ConScape.LF_power_mean_sensitivity(h)[1]
    S_testnodes_pm = S_comp_pm[[h.g.id_to_grid_coordinate_list[i] for i in testnodes]]

    S_simu_pm = ConScape.LF_power_mean_sensitivity_simulation(h, testnodes=testnodes)

    @test maximum(abs.(S_testnodes_pm - S_simu_pm)./abs.(S_testnodes_pm)) < 1e-5

    @test_throws ErrorException ConScape.LF_power_mean_sensitivity_simulation(ConScape.Habitat(g, β=0.5, cost=ConScape.Inv()))
end


@testset "Cost functions" begin
    l = rand(4, 4)

    g = ConScape.Grid(size(l)..., landscape=ConScape.graph_matrix_from_raster(l))

    for c in [ConScape.MinusLog(),
              ConScape.ExpMinus(),
              ConScape.Inv(),
              ConScape.OddsAgainst(),
              ConScape.OddsFor()]

        h_c = ConScape.Habitat(g, β=0.2, cost=c)
        @test h_c isa ConScape.Habitat
    end

    g.A[1,2] = 1.1 # Causes negative cost for C[1,2] when cost=MinusLog
    @test_throws ArgumentError ConScape.Habitat(g, β=0.1, cost=ConScape.MinusLog()) # should raise error, as C[1,2]<0
end

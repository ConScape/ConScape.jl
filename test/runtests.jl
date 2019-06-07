using ConScape, Test, SparseArrays

datadir = joinpath(@__DIR__(), "..", "data")

@testset "Test mean_kl_divergence" begin
    # Create the same landscape in Julia
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
                               # Qualities decrease by row
                               qualities=copy(reshape(collect(1800:-1:1), 60, 30)')
                               )
    h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=0.2)

    @test ConScape.mean_kl_divergence(h) ≈ 31104209170543.438
end

@testset "Read asc data..." begin
    affinities, _ = ConScape.readasc(joinpath(datadir, "affinities.asc"))
    qualities , _ = ConScape.readasc(joinpath(datadir, "qualities.asc"))

    @testset "create Grid" begin
        g = ConScape.Grid(size(affinities)...,
                          landscape=ConScape.adjacency(affinities),
                          qualities=qualities
                          )
        @test g isa ConScape.Grid
    end

    @testset "test adjacency creation with $nn neightbors and $w weighting" for
        nn in (ConScape.N4, ConScape.N8),
            w in (ConScape.TargetWeight, ConScape.AverageWeight)
        # FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
        @test ConScape.adjacency(affinities, neighbors=nn, weight=w) isa ConScape.SparseMatrixCSC
    end
end

@testset "Test betweenness" begin
    affinities, _ = ConScape.readasc(joinpath(datadir, "affinities_sno_2000.asc"))
    qualities , _ = ConScape.readasc(joinpath(datadir, "qualities_sno_2000.asc"))

    g = ConScape.Grid(size(affinities)...,
                          landscape=ConScape.adjacency(affinities),
                          qualities=qualities
                      )
    h = ConScape.Habitat(g, β=0.1)

    # This is a regression test based on values that we currently believe to be correct
    bet = ConScape.RSP_full_betweenness_kweighted(h)
    @test bet[21:23, 31:33] ≈ [0.056248647745559356 0.09283682744933167 0.13009655005263085
                               0.051749956522989624 0.15070574694066693 0.18103463182904647
                               0.0468241782430599   0.2081201353658689  0.29892394108578946]

    # This is a regression test based on values that we currently believe to be correct
    bet = ConScape.RSP_full_betweenness_kweighted(h, invcost=t -> exp(-t/50))
    @test bet[21:23, 31:33] ≈ [1108.2090427345915 1456.7519912636426 1908.1917725150054
                                870.9372313404992 2147.3483997180106 2226.8165679274825
                                770.5051274960429 2573.3261638421927 3434.4832928490296]

    @test ConScape.RSP_full_betweenness_kweighted(h, invcost=one)[g.id_to_grid_coordinate_list] ≈
          ConScape.RSP_full_betweenness_qweighted(h)[g.id_to_grid_coordinate_list]
end

@testset "graph splitting" begin
    l1 = [1/4 0 1/4 1/4
          1/4 0 1/4 1/4
          1/4 0 1/4 1/4
          1/4 0 1/4 1/4]

    l2 = [0   0 1/4 1/4
          0   0 1/4 1/4
          0   0 1/4 1/4
          0   0 1/4 1/4]

    g1 = ConScape.Grid(size(l1)..., landscape=ConScape.adjacency(l1))
    g2 = ConScape.Grid(size(l2)..., landscape=ConScape.adjacency(l2))

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

    g = ConScape.Grid(size(l)..., landscape=ConScape.adjacency(l))

    @test all(ConScape.least_cost_distance(g, (4,4)) .=== [Inf  NaN  0.75  0.75
                                                           Inf  NaN  0.5   0.5
                                                           Inf  NaN  0.25  0.25
                                                           Inf  NaN  0.25  0.0])
end

@testset "custom scaling function in k-weighted betweenness" begin
    l = rand(4, 4)
    q = rand(4, 4)

    g = ConScape.Grid(size(l)..., landscape=ConScape.adjacency(l))
    h = ConScape.Habitat(g, β=0.2)

    @test ConScape.RSP_full_betweenness_kweighted(h) == ConScape.RSP_full_betweenness_kweighted(h; invcost=t -> exp(-t))
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

@testset "RSP_functionality" begin
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
                               qualities=copy(reshape(collect(1800:-1:1), 60, 30)'))
    h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=0.2)

    @test ConScape.ConScape.RSP_functionality(h)[28:30,58:60] ≈ [11230.00960154365 3089.464044624814 77.98295374964728
                                                                12455.993924751103 3512.5483842548506 57.130439979758414
                                                                10677.051974719087 2937.2668774187628 26.29414517656454]'
end

@testset "mean_lc_kl_divergence" begin
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
                               qualities=copy(reshape(collect(1800:-1:1), 60, 30)'))
    h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=0.2)

    @test ConScape.ConScape.mean_lc_kl_divergence(h) ≈ 1.1901061703319367e14
end

using ConScape, Test, SparseArrays

datadir = joinpath(@__DIR__(), "..", "data")

@testset "test landscale: $landscape" for
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
                [sq[10, 15], sq[20, 15], sq[10, 45], sq[20, 45]]))
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
                          landscape=ConScape.adjacency(affinities),
                          qualities=qualities
                          )
        h = ConScape.Habitat(g, β=β)
    end

    @testset "Test mean_kl_divergence" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.mean_kl_divergence(h) ≈ 31104209170543.438
        elseif landscape == "sno_2000" && β == 0.1
            @test ConScape.mean_kl_divergence(h) ≈ 298539.4404998081
        end
    end

    @testset "test adjacency creation with $nn neightbors and $w weighting" for
        nn in (ConScape.N4, ConScape.N8),
            w in (ConScape.TargetWeight, ConScape.AverageWeight)

        if landscape == "sno_2000" && β == 0.1 # No need to test this on sno_100 and doesn't deepend on β
            # FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
            @test ConScape.adjacency(affinities, neighbors=nn, weight=w) isa ConScape.SparseMatrixCSC
        end
    end

    @testset "Test betweenness" begin
        # FIXME Enable all combinations
        if landscape == "sno_2000" && β == 0.1
            # This is a regression test based on values that we currently believe to be correct
            bet = ConScape.RSP_betweenness_kweighted(h)
            @test bet[21:23, 31:33] ≈ [0.056248647745559356 0.09283682744933167 0.13009655005263085
                                       0.051749956522989624 0.15070574694066693 0.18103463182904647
                                       0.0468241782430599   0.2081201353658689  0.29892394108578946]

            # This is a regression test based on values that we currently believe to be correct
            bet = ConScape.RSP_betweenness_kweighted(h, invcost=t -> exp(-t/50))
            @test bet[21:23, 31:33] ≈ [1108.2090427345915 1456.7519912636426 1908.1917725150054
                                        870.9372313404992 2147.3483997180106 2226.8165679274825
                                        770.5051274960429 2573.3261638421927 3434.4832928490296]

            @test ConScape.RSP_betweenness_kweighted(h, invcost=one)[g.id_to_grid_coordinate_list] ≈
                    ConScape.RSP_betweenness_qweighted(h)[g.id_to_grid_coordinate_list]
        end
    end

    @testset "RSP_functionality" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.ConScape.RSP_functionality(h)[28:30,58:60]' ≈
                [11082.654882969266 2664.916100189486 89.420910249988
                 10340.977912804196 2465.918728844169 56.970111157896
                 11119.132467660969 2662.969749775032 33.280379014217]
        end
    end

    @testset "mean_lc_kl_divergence" begin
        # FIXME Enable all combinations
        if landscape == "wall_full" && β == 0.2
            @test ConScape.ConScape.mean_lc_kl_divergence(h) ≈ 1.1901061703319367e14
        elseif landscape == "sno_2000" && β == 0.1
            @test ConScape.ConScape.mean_lc_kl_divergence(h) ≈ 1.5335659790160232e6
        end
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
                    [1.4012984154363496e9 1.3576613474599123e9 1.4013548293211923e9
                     1.7902650081599138e9 2.016569666682126e9  1.790379360572362e9
                     1.3937127669128556e9 1.3505349580912094e9 1.3934168377493184e9]

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

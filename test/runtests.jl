using ConScape, Test

datadir = joinpath(@__DIR__(), "..", "data")

@testset "Test mean_kl_distance" begin
    # Create the same landscape in Julia
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
                               # Qualities decrease by row
                               qualities=copy(reshape(collect(1800:-1:1), 60, 30)')
                               )
    h = ConScape.Habitat(g, cost=ConScape.MinusLog(), β=0.2)

    @test ConScape.mean_kl_distance(h) ≈ 31104209170543.438
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
        # FIXME! Maybee test mean_kl_distance for part of the landscape to make sure they all roughly give the same result
        @test ConScape.adjacency(affinities, neighbors=nn, weight=w) isa ConScape.SparseMatrixCSC
    end
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

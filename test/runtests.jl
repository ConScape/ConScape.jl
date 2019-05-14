using ConScape, Test

datadir = joinpath(@__DIR__(), "..", "data")

@testset "Test mean_kl_distance" begin
    g = ConScape.perm_wall_sim(30, 60, corridorwidths=(3,2), qualities=reshape(collect(1800:-1:1), 30, 60));
    h = ConScape.Habitat(g, ConScape.MinusLog());

    @test ConScape.mean_kl_distance(h, β=0.2) ≈ 31104209170543.438
end

@testset "Read asc data and create Grid" begin
    affinities, _ = ConScape.readasc(joinpath(datadir, "affinities.asc"))
    qualities , _ = ConScape.readasc(joinpath(datadir, "qualities.asc"))
    g = ConScape.Grid(size(affinities)...,
                      landscape=ConScape.adjacency(affinities),
                      qualities=qualities
                      )
    @test g isa ConScape.Grid
end

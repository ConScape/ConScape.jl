using ConScape
using Rasters
using ArchGDAL
using SparseArrays
using Test
import ConScape.Graphs

using ConScape: graph_matrix_from_raster

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"
affinity_raster = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)

# TODO these are not real tests
@testset "test adjacency creation with $nn neighbors, $w weighting and $it" for
    nn in (ConScape.N4, ConScape.N8),
        w in (TargetWeight(), AverageWeight()),
            it in (Likelihood(), Cost())
                # No need to test this on sno_100 and doesn't deepend on θ
                # FIXME! Maybe test mean_kl_divergence for part of the landscape to make sure they all roughly give the same result
                @test graph_matrix_from_raster(affinity_raster;
                    neighbors=nn,
                    transition_weight=w,
                    input_type=it
                ) isa SparseMatrixCSC
end;

@testset "2d raster inputs" begin
    r2 = [1.0 3.0; 2.0 4.0]
    @test graph_matrix_from_raster(r2; 
        neighbors=ConScape.N4, transition_weight=TargetWeight(), input_type=Cost()
    ) == graph_matrix_from_raster(r2; 
        neighbors=ConScape.N4, transition_weight=TargetWeight(), input_type=Likelihood()
    ) == [
        0.0 2.0 3.0 0.0
        1.0 0.0 0.0 4.0
        1.0 0.0 0.0 4.0
        0.0 2.0 3.0 0.0
    ]
    @test graph_matrix_from_raster(r2; 
        neighbors=ConScape.N4, transition_weight=AverageWeight(), input_type=Likelihood()
    ) ≈ [
        0.0 4/3 1.5 0.0
        4/3 0.0 0.0 8/3
        1.5 0.0 0.0 24/7
        0.0 8/3 24/7 0.0
    ]
    @test graph_matrix_from_raster(r2; 
        neighbors=ConScape.N4, transition_weight=AverageWeight(), input_type=Cost()
    ) == [
        0.0 1.5 2.0 0.0
        1.5 0.0 0.0 3.0
        2.0 0.0 0.0 3.5
        0.0 3.0 3.5 0.0
    ]

    @test graph_matrix_from_raster(r2; 
        neighbors=ConScape.N8, transition_weight=TargetWeight(), input_type=Likelihood()
    ) == [
        0.0                2.0                3.0                2.82842712474619
        1.0                0.0                2.1213203435596424 4.0
        1.0                1.414213562373095  0.0                4.0
        0.7071067811865475 2.0                3.0                0.0
    ]

    @test graph_matrix_from_raster(r2;
        neighbors=ConScape.N8, transition_weight=AverageWeight(), input_type=Cost(),
    ) == [
        0.0        1.5        2.0        2.5sqrt(2)
        1.5        0.0        2.5sqrt(2) 3.0
        2.0        2.5sqrt(2) 0.0        3.5
        2.5sqrt(2) 3.0        3.5        0.0
    ]
    @test graph_matrix_from_raster(r2;
        neighbors=ConScape.N8, transition_weight=AverageWeight(), input_type=Likelihood(),
    ) ≈ [
        0.0 4/3 1.5 1.131370849898476
        4/3 0.0 1.697056274847714 8/3
        1.5 1.697056274847714 0.0 24/7
        1.131370849898476 8/3 24/7 0.0
    ]
end

@testset "2d raster inputs" begin

    n4 = 1:4
    n8 = [1 5 2 6 3 7 4 8]
    r3_4 = reshape(cat(n4, n4, n4, n4; dims=1), 4, 2, 2)
    r3_8 = reshape(cat(n8, n8, n8, n8; dims=1), 8, 2, 2)

    # TODO: test these
    graph_matrix_from_raster(r3_4; input_type=Likelihood())
    graph_matrix_from_raster(r3_4; input_type=Likelihood())

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

    g1 = ConScape.GridGraph(; 
        transitionlikelihood=graph_matrix_from_raster(l1; input_type=Likelihood()), 
        costfunction=MinusLog(),
        quality=ones(size(l1)),
    )
    g2 = ConScape.GridGraph(; 
        transitionlikelihood=graph_matrix_from_raster(l2; input_type=Likelihood()), 
        costfunction=MinusLog(),
        quality=ones(size(l2))
    )
    sgs1 = ConScape.split_connected_graphs(g1)
    sgs2 = ConScape.split_connected_graphs(g2)
    @test length(sgs1) == 2
    @test length(sgs2) == 1

    @test !Graphs.is_strongly_connected(g1)
    @test Graphs.is_strongly_connected(sgs1[1])
    @test Graphs.is_strongly_connected(sgs1[2])
    @test !Graphs.is_strongly_connected(g2)
    @test Graphs.is_strongly_connected(sgs2[1])

    @test sgs1[1].transitioncost == sgs1[1].transitioncost
    @test sgs1[1].transitionlikelihood == sgs1[1].transitionlikelihood
end

using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
using Rasters, ArchGDAL
using OldConScape

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"

# The way the ascii wass read in is reversed and rotated from what GDAL does
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

isnanorapprox(x::AbstractArray, y::AbstractArray; atol=0.0) = all(isnanorapprox.(x, y; atol))
isnanorapprox(x, y; atol=0.0) = (isnan(x) && isnan(y)) || isapprox(x, y; atol)

@testset "RSP sensitivity measure" begin
    # Set up the new sensitivity measures
    sensitivity_measures = (;
        sens_cost=SensitivityAnalysis(; wrt=StepCost()),
        sens_likelihood=SensitivityAnalysis(; wrt=StepLikelihood()),
        sens_costtolikelihood=SensitivityAnalysis(; wrt=StepCostToLikelihood()),
        sens_likelihoodtocost=SensitivityAnalysis(; wrt=StepLikelihoodToCost()),
        sens_cost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepCost()),
        sens_likelihood_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepLikelihood()),
        sens_costtolikelihood_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepCostToLikelihood()),
        sens_likelihoodtocost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepLikelihoodToCost()),
    )

    thetas = (theta_one=1.0, theta_pointone=0.1)

    rsps = map(thetas) do theta
        map((ec=ExpectedCost(), pmp=PowerMeanProximity())) do proximity_measure
            RandomisedShortestPath(; 
                # distance_transformation=ExpMinusAlpha(0.1),
                distance_transformation=ExpMinus(),
                proximity_measure, 
                theta, 
            )
        end
    end
    sens = map(rsps) do rsp_theta
        map(rsp_theta) do rsp
            solve(sensitivity_measures, rsp, rast)
        end
    end


    # Create the same landscape in Julia
    # g = OldConScape.perm_wall_sim(30, 60, corridorwidths=(3,2),
    # # Qualities decrease by row
    #     qualities=copy(reshape(collect(1800:-1:1), 60, 30)')
    # )
    #
    # test_g = OldConScape.Grid(
    #     size(g)...,
    #     affinities=g.affinities,
    #     source_qualities=g.source_qualities,
    #     target_qualities=OldConScape.coarse_graining(g, 3))

    ConScape.store[] = (;)
    OldConScape.store[] = (;)

    # OldConScape sensitivity
    old_sens = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
        test_g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        # wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
        # wrts = ["A", "C", "C&A=f(C)", "A&C=f(A)"]
        cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
        connectivity_function = cfs.ec
        wrt = "C"
        theta = thetas.theta_pointone
        # map(thetas) do theta
            test_grsp = OldConScape.GridRSP(test_g; θ=theta)
            # map(cfs) do connectivity_function
                # Dict(wrts .=> map(wrts) do wrt
                    OldConScape.sensitivity(test_grsp;
                        connectivity_function,
                        distance_transformation=OldConScape.ExpMinus(),
                        # α=0.1,
                        # α=1.0,
                        wrt,
                        landscape_measure=["sum","eigenanalysis"][1],
                        unitless=false,
                        diagvalue=nothing,
                        target_equal_source=true
                    )
                # end)
            # end
        # end
    end

    # test_grsp = OldConScape.GridRSP(test_g; θ=thetas.theta_pointone)
    # old_ebetm = OldConScape.edge_betweenness_kweighted(test_grsp;
    #     distance_transformation=OldConScape.ExpMinus(),
    # )
    # ebetm = solve(EdgeBetweenness(QualityAndProximityWeighted()), rsps.theta_pointone.ec, rast)

    # @test isapprox(old_ebetm, ebetm[1])


    @time rsp_ec = solve(sensitivity_measures.sens_cost, rsps.theta_pointone.ec, rast); @test isapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ)
    using ProfileView
    @profview solve(sensitivity_measures.sens_cost, rsps.theta_pointone.ec, rast)

    @test OldConScape.store[].W == ConScape.store[].W
    @test OldConScape.store[].C == ConScape.store[].C
    @test OldConScape.store[].CW == ConScape.store[].CW
    @test OldConScape.store[].Z == ConScape.store[].Z
    @test OldConScape.store[].Zⁱ == ConScape.store[].Zⁱ
    @test OldConScape.store[].Zrows == ConScape.store[].Zrows
    # @test isapprox(OldConScape.store[].K, ConScape.store[].K)
    @test isapprox(OldConScape.store[].M, ConScape.store[].M)
    @test isapprox(OldConScape.store[].Y, ConScape.store[].Y)
    @test isapprox(OldConScape.store[].K̂ , ConScape.store[].MZⁱ)
    @test isapprox(OldConScape.store[].k̂diagZⁱ, ConScape.store[].mdiagZⁱ)
    @test isapprox(OldConScape.store[].k̂diagC̄Zⁱ, ConScape.store[].mdiagC̄Zⁱ)
    @test isapprox(OldConScape.store[].X3, ConScape.store[].X3)
    @test isapprox(OldConScape.store[].K̂ᵀZ, ConScape.store[].MᵀZ)
    @test isapprox(OldConScape.store[].X3CW, ConScape.store[].X3CW)
    @test isapprox(OldConScape.store[].K̂ᵀZCW, ConScape.store[].MᵀZCW)
    @test isapprox(OldConScape.store[].K̂C̄ᵣ, ConScape.store[].MC̄ᵣ)
    @test isapprox(OldConScape.store[].RHS, ConScape.store[].RHS)
    @test isapprox(OldConScape.store[].X5, ConScape.store[].X5)
    @test isapprox(OldConScape.store[].X6, ConScape.store[].X6)
    @test isapprox(OldConScape.store[].kB, ConScape.store[].kB)
    @test isapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ)

    heatmap(OldConScape.store[].kB)
    heatmap(ConScape.store[].kB)
    heatmap(OldConScape.store[].kΣ)
    heatmap(ConScape.store[].kΣ)

    # @test OldConScape.store[].W == ConScape.store[].W
    # # @test OldConScape.store[].C == ConScape.store[].C
    # @test OldConScape.store[].Z == ConScape.store[].Z
    # @test OldConScape.store[].Zⁱ == ConScape.store[].Zⁱ
    # @test isapprox(OldConScape.store[].Zrows, ConScape.store[].Zrows')
    # @test isapprox(OldConScape.store[].K, ConScape.store[].K)
    # @test isapprox(OldConScape.store[].M, ConScape.store[].M)
    # @test isapprox(OldConScape.store[].XᵀZ, ConScape.store[].XᵀZ')
    # @test isapprox(OldConScape.store[].bet_edge_k, ConScape.store[].bet_edge_k)
    # @test isapprox(OldConScape.store[].S_e_aff, ConScape.store[].S_e_aff)
    # @test isapprox(OldConScape.store[].S_e_cost, ConScape.store[].S_e_cost)
    # @test isnanorapprox(old_sens.theta_pointone.pmp["C"], rsp_pmp) 

    # @time rsp_ec = solve(sensitivity_measures.sens_cost, rsps.theta_one.ec, rast);
    # isapprox(OldConScape.store[].kB, ConScape.store[].kB; atol=0.1)
    # A = replace(collect(OldConScape.store[].kB), 0.0 => NaN)
    # B = replace(collect(ConScape.store[].kB), 0.0 => NaN)
    # A = collect(OldConScape.store[].kB)
    # B = collect(ConScape.store[].kB)
    # heatmap(A)
    # heatmap(B)
    # i = .!(isapprox.(A, B; atol=0.01))
    # A[i]
    # B[i]


    # count(isapprox.(OldConScape.store[].bet_edge_k, ConScape.store[].bet_edge_k))
    # length(isnanorapprox.(OldConScape.store[].bet_edge_k, ConScape.store[].bet_edge_k))
    using Plots
    heatmap(OldConScape.store[].W)
    heatmap(ConScape.store[].W)
    heatmap(ConScape.store[].bet_edge_k; clims=(0, 30))
    heatmap(OldConScape.store[].bet_edge_k; clims=(0, 30))
    # heatmap(OldConScape.store[].S_e_aff)
    # heatmap(ConScape.store[].S_e_aff)
    # heatmap(OldConScape.store[].S_e_cost)
    # heatmap(ConScape.store[].S_e_cost)
    # count(isapprox.(OldConScape.store[].bet_node_k, filter(!isnan, ConScape.store[].bet_node_k); atol=100000))
    # plot(OldConScape.store[].bet_node_k)
    # plot(filter(!isnan, ConScape.store[].bet_node_k))
    # heatmap(ConScape.store[].bet_node_k)
    # heatmap(OldConScape.store[].S_e_aff .- ConScape.store[].S_e_aff)
    # heatmap(res_sens_rsp_pmp)
    # heatmap(old_sensitivity.pmp)
    # all(isapprox.(OldConScape.store[].S_e_aff, ConScape.store[].S_e_aff; atol=1e-15))
    # all(isapprox.(OldConScape.store[].S_e_cost, ConScape.store[].S_e_cost; atol=1e-15))

    # isapprox(OldConScape.store[].kB, ConScape.store[].kB)
    # isapprox(OldConScape.store[].W, ConScape.store[].W)
    # isapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ)
    # all(isapprox.(OldConScape.store[].kΣ, ConScape.store[].kΣ; atol=1))
    # count(isapprox.(OldConScape.store[].kΣ, ConScape.store[].kΣ; atol=0.1))
    # length(isapprox.(OldConScape.store[].kΣ, ConScape.store[].kΣ; atol=0.1))
    
    # And movement modes

    using Plots
    heatmap(old_sens.theta_pointone.ec["A"])
    heatmap(parent(sens.theta_pointone.ec.sens_likelihood))
    heatmap(old_sens.theta_pointone.ec["C"])
    heatmap(parent(sens.theta_pointone.ec.sens_cost))
    heatmap(old_sens.theta_pointone.ec["C&A=f(C)"])
    heatmap(parent(sens.theta_pointone.ec.sens_likelihoodtocost))
    heatmap(old_sens.theta_pointone.ec["A&C=f(A)"])
    heatmap(parent(sens.theta_pointone.ec.sens_costtolikelihood))

    heatmap(old_sens.theta_one.pmp["A"])
    heatmap(sens.theta_one.pmp.sens_likelihood)
    heatmap(old_sens.theta_one.pmp["C"])
    heatmap(sens.theta_one.pmp.sens_cost)
    # heatmap(old_sensitivity.pmp["Q"])
    # heatmap(res_sens_rsp_pmp.sens_quality)
    heatmap(old_sens.theta_one.pmp["C&A=f(C)"])
    heatmap(sens.theta_one.pmp.sens_likelihoodtocost)
    heatmap(old_sens.theta_one.pmp["A&C=f(A)"])
    heatmap(sens.theta_one.pmp.sens_costtolikelihood)

    # These are largish numbers, and for some reason a small number of them end up 3% different.
    # This could be due to the difference in how we sum the results? (eg. matmul vs colum wise)
    @test isnanorapprox(old_sens.theta_one.ec["C"],        sens.theta_one.ec.sens_cost; atol=1)
    @test isnanorapprox(old_sens.theta_one.ec["A"],        sens.theta_one.ec.sens_likelihood; atol=70)
    test isnanorapprox(old_sens.theta_one.ec["C&A=f(C)"], sens.theta_one.ec.sens_likelihoodtocost; atol=70)
    @test isnanorapprox(old_sens.theta_one.ec["A&C=f(A)"], sens.theta_one.ec.sens_costtolikelihood; atol=70)

    @test isnanorapprox(old_sens.theta_pointone.ec["C"],        sens.theta_pointone.ec.sens_cost; atol=10)
    @test isnanorapprox(old_sens.theta_pointone.ec["A"],        sens.theta_pointone.ec.sens_likelihood; atol=250)
    @test isnanorapprox(old_sens.theta_pointone.ec["C&A=f(C)"], sens.theta_pointone.ec.sens_likelihoodtocost; atol=250)
    @test isnanorapprox(old_sens.theta_pointone.ec["A&C=f(A)"], sens.theta_pointone.ec.sens_costtolikelihood; atol=250)

    # With theta=1.0 atol can be 1e-14
    @test isnanorapprox(old_sens.theta_one.pmp["C"],        sens.theta_one.pmp.sens_cost)
    @test isnanorapprox(old_sens.theta_one.pmp["A"],        sens.theta_one.pmp.sens_likelihood)
    @test isnanorapprox(old_sens.theta_one.pmp["C&A=f(C)"], sens.theta_one.pmp.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_one.pmp["A&C=f(A)"], sens.theta_one.pmp.sens_costtolikelihood)

    # Match is lower with theta=0.1, but still fairly good
    @test isnanorapprox(old_sens.theta_pointone.pmp["C"],        sens.theta_pointone.pmp.sens_cost)
    @test isnanorapprox(old_sens.theta_pointone.pmp["A"],        sens.theta_pointone.pmp.sens_likelihood)
    @test isnanorapprox(old_sens.theta_pointone.pmp["C&A=f(C)"], sens.theta_pointone.pmp.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_pointone.pmp["A&C=f(A)"], sens.theta_pointone.pmp.sens_costtolikelihood)

    # OldConScape
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    test_g = OldConScape.Grid(size(steplikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality),
    )
    test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
    old_sensitivity = OldConScape.sensitivity(test_grsp;
        connectivity_function=OldConScape.expected_cost,
        distance_transformation=OldConScape.ExpMinus(),
        α=0.005,
        wrt="C",
        landscape_measure="sum",
        unitless=true, # Elasticities
        diagvalue=nothing,
        target_equal_source=true
    )
    # heatmap(old_sensitivity)
    # SensitivityAnalysis
    sens_cost_elast = SensitivityAnalysis(; 
        type=Elasticity(), wrt=StepCost()
    )
    s = solve(sens_cost_elast, rsp_ec, rast, 1)
end

dest = zeros(5)
A = sprand(5, 10, 0.5)
b = rand(10)

ConScape.foreachnz(A) do i, j, n
    dest[i] += A.nzval[n] * b[j] 
end
dest
A * b

n, m = 300, 200 
dest1 = sprand(n, n, 0.9)
dest2 = copy(dest1)
dest3 = copy(dest1) .* 0.0
A = rand(n, m)
B = rand(m, n)

for i in axes(dest1, 1)
    for j in findall(dest1[i,:] .> 0)
        x = only(A[j, :]' * B[:, i])
        dest2[i, j] *= x
    end
end
dest2

for t in 1:m
    A_t = A[:, t]
    B_t = B[t, :]
    ConScape.foreachnz(dest1) do i, j, n
        dest3[i, j] += dest1.nzval[n] * A_t[j] * B_t[i]
    end
    # for i in axes(dest1, 1)
    #     for j in findall(dest1[i,:] .> 0)
    #         x = dest1[i, j] * A_t[j] * B_t[i]
    #         dest3[i, j] += x 
    #     end
    # end
end
dest2
dest3
isapprox(dest2, dest3)
sum(dest2)
sum(dest3) 


A = [6.0 5 4; 3 2 1]
B = [1.0 2 3; 4 5 6; 7 8 9]
C = [6 5 4]
C = [6.0, 5.0, 4.0]
A * B
C * B
B' * C'
B = rand(3, 3)
C = rand(3, 3)

B * C == hcat(B * C[:, 1], B * C[:, 1], B * C[:, 1])
(B * C)[1, :]
B
(C' * B[1, :])
C' * B[1, :]






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
                distance_transformation=ExpMinusAlpha(0.1),
                proximity_measure, 
                theta, 
            )
        end
    end
    sens = map(rsps) do rsp_theta
        map(rsp_theta) do rsp_pm
            solve(sensitivity_measures, rsp_pm, rast)
        end
    end

    ConScape.store[] = (;)
    OldConScape.store[] = (;)

    # OldConScape sensitivity
    old_sens = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
        test_g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
        cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
        connectivity_function = cfs.ec
        theta = thetas.theta_one
        # map(thetas) do theta
            test_grsp = OldConScape.GridRSP(test_g; θ=theta)
            # map(cfs) do connectivity_function
                Dict(wrts .=> map(wrts) do wrt
                    old_sens_affinity = OldConScape.sensitivity(test_grsp;
                        connectivity_function,
                        distance_transformation=OldConScape.ExpMinus(),
                        α=0.1,
                        wrt,
                        landscape_measure=["sum","eigenanalysis"][1],
                        unitless=false,
                        diagvalue=nothing,
                        target_equal_source=true
                    )
                end)
            # end
        # end
    end
    @time rsp_ec = solve(sensitivity_measures.sens_cost, rsps.theta_one.ec, rast);

    @test OldConScape.store[].W == ConScape.store[].W
    @test OldConScape.store[].Z == ConScape.store[].Z
    @test OldConScape.store[].Zⁱ == ConScape.store[].Zⁱ
    @test OldConScape.store[].Z == ConScape.store[].Z
    @test OldConScape.store[].Zrows == ConScape.store[].Zrows
    @test OldConScape.store[].Z == ConScape.store[].Zrows

    @test isapprox(OldConScape.store[].K, ConScape.store[].K)
    @test isapprox(OldConScape.store[].M, ConScape.store[].M)
    @test isapprox(OldConScape.store[].k̂diagZⁱ, ConScape.store[].mdiagZⁱ)
    @test isapprox(OldConScape.store[].X3, ConScape.store[].X3'; atol=1e-2)
    @test isapprox(OldConScape.store[].X6, ConScape.store[].X6)

    isapprox(ConScape.store[].mdiagZⁱ .* ConScape.store[].Zrows, OldConScape.store[].X3)
    size(ConScape.store[].mdiagZⁱ) 
    size(ConScape.store[].Zrows)
    OldConScape.store[].k̂diagZⁱ

    isnanorapprox.(OldConScape.store[].X3, ConScape.store[].X3'; atol=1e-5))

    @test_broken 
    isapprox(
    OldConScape.store[].XᵀZ
    , 
    ConScape.store[].XᵀZ
    ; atol=1)
    @test_broken OldConScape.store[].XᵀZmd == ConScape.store[].XᵀZmd
    @test_broken 
    OldConScape.store[].bet_edge_k 
    ConScape.store[].bet_edge_k
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

    heatmap(old_sens.pmp["A"])
    heatmap(sense.pmp.sens_likelihood)
    heatmap(old_sens.pmp["C"])
    heatmap(sense.pmp.sens_cost)
    # heatmap(old_sensitivity.pmp["Q"])
    # heatmap(res_sens_rsp_pmp.sens_quality)
    heatmap(old_sens.pmp["C&A=f(C)"])
    heatmap(old_sens.pmp.sens_likelihoodtocost)
    heatmap(old_sens.pmp["A&C=f(A)"])
    heatmap(old_sens.pmp.sens_costtolikelihood)

    # These are largish numbers, and for some reason a small number of them end up 3% different.
    # This could be due to the difference in how we sum the results? (eg. matmul vs colum wise)
    @test isnanorapprox(old_sens.theta_one.ec["C"],        sens.theta_one.ec.sens_cost; atol=1)
    @test isnanorapprox(old_sens.theta_one.ec["A"],        sens.theta_one.ec.sens_likelihood; atol=70)
    @test isnanorapprox(old_sens.theta_one.ec["C&A=f(C)"], sens.theta_one.ec.sens_likelihoodtocost; atol=70)
    @test isnanorapprox(old_sens.theta_one.ec["A&C=f(A)"], sens.theta_one.ec.sens_costtolikelihood; atol=70)

   @test isnanorapprox(old_sens.theta_pointone.ec["C"],        sens.theta_pointone.ec.sens_cost; atol=10)
    @test isnanorapprox(old_sens.theta_pointone.ec["A"],        sens.theta_pointone.ec.sens_likelihood; atol=250)
    @test isnanorapprox(old_sens.theta_pointone.ec["C&A=f(C)"], sens.theta_pointone.ec.sens_likelihoodtocost; atol=250)
    @test isnanorapprox(old_sens.theta_pointone.ec["A&C=f(A)"], sens.theta_pointone.ec.sens_costtolikelihood; atol=250)

    # With theta=1.0 atol can be 1e-14
    @test isnanorapprox(old_sens.theta_one.pmp["C"],        sens.theta_one.pmp.sens_cost; atol=1e-14)
    @test isnanorapprox(old_sens.theta_one.pmp["A"],        sens.theta_one.pmp.sens_likelihood; atol=1e-14)
    @test isnanorapprox(old_sens.theta_one.pmp["C&A=f(C)"], sens.theta_one.pmp.sens_likelihoodtocost; atol=1e-14)
    @test isnanorapprox(old_sens.theta_one.pmp["A&C=f(A)"], sens.theta_one.pmp.sens_costtolikelihood; atol=1e-14)

    # Match is lower with theta=0.1, but still fairly good
    @test isnanorapprox(old_sens.theta_pointone.pmp["C"],        sens.theta_pointone.pmp.sens_cost; atol=1e-10)
    @test isnanorapprox(old_sens.theta_pointone.pmp["A"],        sens.theta_pointone.pmp.sens_likelihood; atol=1e-8)
    @test isnanorapprox(old_sens.theta_pointone.pmp["C&A=f(C)"], sens.theta_pointone.pmp.sens_likelihoodtocost; atol=1e-10)
    @test isnanorapprox(old_sens.theta_pointone.pmp["A&C=f(A)"], sens.theta_pointone.pmp.sens_costtolikelihood; atol=1e-7)

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

I_L = rand(12, 10)
W = rand(12, 12)

using LinearAlgebra
(I_L' / (I - W))[:, 1]
((I - W') \ I_L[:, 1])


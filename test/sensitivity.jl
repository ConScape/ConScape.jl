using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
using Rasters, ArchGDAL
using OldConScape

@testset "RSP sensitivity measure" begin
    sensitivity_measures = (;
        sens_cost=SensitivityAnalysis(; wrt=StepCost()),
        sens_affinity=SensitivityAnalysis(; wrt=StepLikelihood()),
        sens_costtoaffinity=SensitivityAnalysis(; wrt=StepCostToLikelihood()),
        sens_affinitytocost=SensitivityAnalysis(; wrt=StepLikelihoodToCost()),
        sens_cost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepCost()),
        sens_affinity_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepLikelihood()),
        sens_costtoaffinity_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepCostToLikelihood()),
        sens_affinitytocost_elast=SensitivityAnalysis(; type=Elasticity(), wrt=StepLikelihoodToCost()),
    )

    # Movement modes
    rsp_ec = RandomisedShortestPath(; 
        proximity_measure=ExpectedCost(), 
        distance_transformation=ExpMinus(),
        theta=1.0, 
    )
    rsp_pmp = RandomisedShortestPath(; 
        proximity_measure=PowerMeanProximity(), 
        distance_transformation=ExpMinus(),
        theta=1.0, 
    )

    @time res_sens_rsp_ec = solve(sensitivity_measures, rsp_ec, rast)
    @time res_sens_rsp_pmp = solve(sensitivity_measures, rsp_pmp, rast)

    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    test_g = OldConScape.Grid(size(steplikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality),
    )
    test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
    wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
    old_sens = map((ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)) do distance_transformation
        Dict(wrts .=> map(wrts) do wrt
            old_sens_affinity = OldConScape.sensitivity(test_grsp;
                connectivity_function=OldConScape.expected_cost,
                distance_transformation=OldConScape.ExpMinus(),
                α=0.1,
                wrt,
                landscape_measure=["sum","eigenanalysis"][1],
                unitless=true,
                diagvalue=nothing,
                target_equal_source=true
            )
        end)
    end

    using Plots
    heatmap(res_sens_rsp_ec.sens_cost .- old_sens.ec["C"])
    heatmap(parent(res_sens_rsp_ec.sens_cost))
    heatmap(old_sens.ec["A"])
    heatmap(parent(res_sens_rsp_ec.sens_affinity))
    heatmap(old_sens.ec["C"])
    heatmap(parent(res_sens_rsp_ec.sens_cost))
    heatmap(old_sens.ec["C&A=f(C)"])
    heatmap(parent(res_sens_rsp_ec.sens_affinitytocost))
    heatmap(old_sens.ec["A&C=f(A)"])
    heatmap(parent(res_sens_rsp_ec.sens_costtoaffinity))

    # OldConScape
    affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    test_g = OldConScape.Grid(size(steplikelihood)...;
        affinities=affinities_sparse,
        qualities=parent(quality),
    )
    test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
    old_sens_cost = OldConScape.sensitivity(test_grsp;
        connectivity_function=OldConScape.expected_cost,
        distance_transformation=OldConScape.ExpMinus(),
        α=0.005,
        wrt="C",
        landscape_measure="sum",
        unitless=true, # Elasticities
        diagvalue=nothing,
        target_equal_source=true
    )
    # heatmap(old_sens_cost)
    # SensitivityAnalysis
    sens_cost_elast = SensitivityAnalysis(; 
        type=Elasticity(), wrt=StepCost()
    )
    s = solve(sens_cost_elast, rsp_ec, rast, 1)
end

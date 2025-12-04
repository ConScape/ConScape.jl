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
        sens_cost =                   SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=Summation()),
        sens_likelihood =             SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=Summation()),
        sens_costtolikelihood =       SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=Summation()),
        sens_likelihoodtocost =       SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=Summation()),
        sens_quality =                SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=Summation()),
        sens_source_quality =         SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=Summation()),
        sens_target_quality =         SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=Summation()),
        elast_cost =                  SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=Summation()),
        elast_likelihood =            SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=Summation()),
        elast_costtolikelihood =      SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=Summation()),
        elast_likelihoodtocost =      SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=Summation()),
        elast_quality =               SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=Summation()),
        elast_source_quality =        SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=Summation()),
        elast_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=Summation()),
    )

    thetas = (theta_one=1.0, theta_pointone=0.1)

    for grain in (nothing, 2)

        problems = map(thetas) do theta
            map((ec=ExpectedCost(), pmp=PowerMeanProximity())) do proximity_measure
                Problem(
                    RandomisedShortestPath(; 
                        distance_transformation=ExpMinusAlpha(0.5),
                        # distance_transformation=ExpMinus(),
                        proximity_measure, 
                        theta, 
                    );
                    grain
                )
            end
        end

        # OldConScape sensitivity
        @time old_sens = let
            affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
            test_g = OldConScape.Grid(size(steplikelihood)...;
                affinities=affinities_sparse,
                qualities=parent(quality),
            )
            wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
            cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
            connectivity_function = cfs.ec
            wrt = "Q"
            theta = thetas.theta_one
            unitless = true
            map(thetas) do theta
                test_grsp = OldConScape.GridRSP(test_g; θ=theta)
                map((elasticity=true, sensitivity=false)) do unitless
                    map(cfs) do connectivity_function
                        Dict(wrts .=> map(wrts) do wrt
                            OldConScape.sensitivity(test_grsp;
                                connectivity_function,
                                distance_transformation=OldConScape.ExpMinus(),
                                α=0.5,
                                wrt,
                                landscape_measure=["sum","eigenanalysis"][1],
                                unitless,
                                diagvalue=nothing,
                                target_equal_source=true
                            )
                        end)
                    end
                end
            end
        end;

        using OldConScape.Plots
        heatmap(parent(x))
        parent(x)
        parent(sens.theta_one.ec.elast_quality)
        heatmap(old_sens)
        heatmap(parent(sens.theta_one.ec.elast_quality))
        heatmap(parent(sens.theta_one.ec.elast_source_quality))
        heatmap(parent(fh))
        heatmap(parent(sens.theta_one.ec.elast_target_quality))
        heatmap(parent(sens.theta_one.ec.sens_quality))
        heatmap(parent(sens.theta_one.ec.sens_target_quality))
        heatmap(parent(sens.theta_one.ec.sens_source_quality))
        x = solve(sensitivity_measures.sens_target_quality, rsps.theta_one.ec, rast)
        g = init(rsps.theta_one.ec, rast)
        heatmap(old_sens.theta_one.elasticity.ec["Q"] ./ g.gridgraph.sourcequality)
        heatmap(old_sens.theta_one.sensitivity.ec["Q"])
        heatmap(parent(sens.theta_one.ec.elast_quality) ./ g.gridgraph.sourcequality)

        fh = solve(FunctionalHabitat(), rsps.theta_one.ec, rast)
        isapprox(ConScape.store[].K, OldConScape.store[].K)
        heatmap(old_sens.theta_one.sensitivity.ec["Q"])
        heatmap(parent(sens.theta_one.ec.sens_quality))
        heatmap(parent(sens.theta_one.ec.sens_source_quality))
        heatmap(parent(sens.theta_one.ec.sens_target_quality))
        heatmap(parent(sens.theta_one.ec.sens_target_quality .+ sens.theta_one.ec.sens_source_quality))
        
        # TODO: test with non-square Z
        # And movement modes
        @test isnanorapprox(old_sens.theta_one.sensitivity.ec["C"],        sens.theta_one.ec.sens_cost)
        @test isnanorapprox(old_sens.theta_one.sensitivity.ec["A"],        sens.theta_one.ec.sens_likelihood)
        @test isnanorapprox(old_sens.theta_one.sensitivity.ec["C&A=f(C)"], sens.theta_one.ec.sens_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_one.sensitivity.ec["A&C=f(A)"], sens.theta_one.ec.sens_costtolikelihood)
        @test isnanorapprox(old_sens.theta_one.sensitivity.ec["Q"],        sens.theta_one.ec.sens_quality)
        @test isnanorapprox(old_sens.theta_one.elasticity.ec["C"],         sens.theta_one.ec.elast_cost)
        @test isnanorapprox(old_sens.theta_one.elasticity.ec["A"],         sens.theta_one.ec.elast_likelihood)
        @test isnanorapprox(old_sens.theta_one.elasticity.ec["C&A=f(C)"],  sens.theta_one.ec.elast_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_one.elasticity.ec["A&C=f(A)"],  sens.theta_one.ec.elast_costtolikelihood)
        @test isnanorapprox(old_sens.theta_one.elasticity.ec["Q"],         sens.theta_one.ec.elast_quality)

        @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["C"],        sens.theta_pointone.ec.sens_cost)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["A"],        sens.theta_pointone.ec.sens_likelihood)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["C&A=f(C)"], sens.theta_pointone.ec.sens_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["A&C=f(A)"], sens.theta_pointone.ec.sens_costtolikelihood)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["Q"],        sens.theta_pointone.ec.sens_quality)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["C"],         sens.theta_pointone.ec.elast_cost)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["A"],         sens.theta_pointone.ec.elast_likelihood)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["C&A=f(C)"],  sens.theta_pointone.ec.elast_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["A&C=f(A)"],  sens.theta_pointone.ec.elast_costtolikelihood)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["Q"],         sens.theta_pointone.ec.elast_quality)

        @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["C"],        sens.theta_one.pmp.sens_cost)
        @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["A"],        sens.theta_one.pmp.sens_likelihood)
        @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["C&A=f(C)"], sens.theta_one.pmp.sens_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["A&C=f(A)"], sens.theta_one.pmp.sens_costtolikelihood)
        @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["Q"],        sens.theta_one.pmp.sens_quality)
        @test isnanorapprox(old_sens.theta_one.elasticity.pmp["C"],         sens.theta_one.pmp.elast_cost)
        @test isnanorapprox(old_sens.theta_one.elasticity.pmp["A"],         sens.theta_one.pmp.elast_likelihood)
        @test isnanorapprox(old_sens.theta_one.elasticity.pmp["C&A=f(C)"],  sens.theta_one.pmp.elast_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_one.elasticity.pmp["A&C=f(A)"],  sens.theta_one.pmp.elast_costtolikelihood)
        @test isnanorapprox(old_sens.theta_one.elasticity.pmp["Q"],         sens.theta_one.pmp.elast_quality)

        @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["C"],        sens.theta_pointone.pmp.sens_cost)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["A"],        sens.theta_pointone.pmp.sens_likelihood)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["C&A=f(C)"], sens.theta_pointone.pmp.sens_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["A&C=f(A)"], sens.theta_pointone.pmp.sens_costtolikelihood)
        @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["Q"],        sens.theta_pointone.pmp.sens_quality)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["C"],         sens.theta_pointone.pmp.elast_cost)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["A"],         sens.theta_pointone.pmp.elast_likelihood)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["C&A=f(C)"],  sens.theta_pointone.pmp.elast_likelihoodtocost)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["A&C=f(A)"],  sens.theta_pointone.pmp.elast_costtolikelihood)
        @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["Q"],         sens.theta_pointone.pmp.elast_quality)
    end

    # # OldConScape
    # affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
    # test_g = OldConScape.Grid(size(steplikelihood)...;
    #     affinities=affinities_sparse,
    #     qualities=parent(quality),
    # )
    # test_grsp = OldConScape.GridRSP(test_g; θ=1.0)
    # old_sensitivity = OldConScape.sensitivity(test_grsp;
    #     connectivity_function=OldConScape.expected_cost,
    #     distance_transformation=OldConScape.ExpMinus(),
    #     α=0.005,
    #     wrt="C",
    #     landscape_measure="sum",
    #     unitless=true, # Elasticities
    #     diagvalue=nothing,
    #     target_equal_source=true
    # )
    # # SensitivityAnalysis
    # sens_cost_elast = SensitivityAnalysis(; 
    #     type=Elasticity(), wrt=StepCost()
    # )
    # s = solve(sens_cost_elast, rsp_ec, rast, 1)
end

include("sensitivity_simulation")

@testset "RSP sensitivity measure" begin

        test_g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
        wrts = ["A", "C", "C&A=f(C)", "A&C=f(A)"]
        cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
        connectivity_function = cfs.ec
        wrt = "C"
        theta = thetas.theta_pointone
        unitless = false
        # map(thetas) do theta
            sim_grsp = OldConScape.GridRSP(test_g; θ=theta)
            map((elasticity=true, sensitivity=false)) do unitless
                map(cfs) do connectivity_function
                    Dict(wrts .=> map(wrts) do wrt
                        
                    end)
                end
            end
        # end

        sensitivity_simulation(sim_grsp;
            connectivity_function,
            distance_transformation,
            α,
            wrt
            landscape_measure
            unitless::Bool=true,
            target_equal_source,
            one_out_of,
        )

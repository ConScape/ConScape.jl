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
                distance_transformation=ExpMinusAlpha(0.5),
                # distance_transformation=ExpMinus(),
                proximity_measure, 
                theta, 
            )
        end
    end

    @time sens = map(rsps) do rsp_theta
        map(rsp_theta) do rsp
            solve(sensitivity_measures, rsp, rast)
        end
    end;

    # OldConScape sensitivity
    @time old_sens = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
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
    
    # And movement modes
    @test isnanorapprox(old_sens.theta_one.sensitivity.ec["C"],        sens.theta_one.ec.sens_cost)
    @test isnanorapprox(old_sens.theta_one.sensitivity.ec["A"],        sens.theta_one.ec.sens_likelihood)
    @test isnanorapprox(old_sens.theta_one.sensitivity.ec["C&A=f(C)"], sens.theta_one.ec.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_one.sensitivity.ec["A&C=f(A)"], sens.theta_one.ec.sens_costtolikelihood)
    @test isnanorapprox(old_sens.theta_one.elasticity.ec["C"],         sens.theta_one.ec.sens_cost_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.ec["A"],         sens.theta_one.ec.sens_likelihood_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.ec["C&A=f(C)"],  sens.theta_one.ec.sens_likelihoodtocost_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.ec["A&C=f(A)"],  sens.theta_one.ec.sens_costtolikelihood_elast)

    @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["C"],        sens.theta_pointone.ec.sens_cost)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["A"],        sens.theta_pointone.ec.sens_likelihood)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["C&A=f(C)"], sens.theta_pointone.ec.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.ec["A&C=f(A)"], sens.theta_pointone.ec.sens_costtolikelihood)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["C"],         sens.theta_pointone.ec.sens_cost_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["A"],         sens.theta_pointone.ec.sens_likelihood_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["C&A=f(C)"],  sens.theta_pointone.ec.sens_likelihoodtocost_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.ec["A&C=f(A)"],  sens.theta_pointone.ec.sens_costtolikelihood_elast)

    @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["C"],        sens.theta_one.pmp.sens_cost)
    @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["A"],        sens.theta_one.pmp.sens_likelihood)
    @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["C&A=f(C)"], sens.theta_one.pmp.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_one.sensitivity.pmp["A&C=f(A)"], sens.theta_one.pmp.sens_costtolikelihood)
    @test isnanorapprox(old_sens.theta_one.elasticity.pmp["C"],         sens.theta_one.pmp.sens_cost_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.pmp["A"],         sens.theta_one.pmp.sens_likelihood_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.pmp["C&A=f(C)"],  sens.theta_one.pmp.sens_likelihoodtocost_elast)
    @test isnanorapprox(old_sens.theta_one.elasticity.pmp["A&C=f(A)"],  sens.theta_one.pmp.sens_costtolikelihood_elast)

    @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["C"],        sens.theta_pointone.pmp.sens_cost)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["A"],        sens.theta_pointone.pmp.sens_likelihood)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["C&A=f(C)"], sens.theta_pointone.pmp.sens_likelihoodtocost)
    @test isnanorapprox(old_sens.theta_pointone.sensitivity.pmp["A&C=f(A)"], sens.theta_pointone.pmp.sens_costtolikelihood)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["C"],         sens.theta_pointone.pmp.sens_cost_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["A"],         sens.theta_pointone.pmp.sens_likelihood_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["C&A=f(C)"],  sens.theta_pointone.pmp.sens_likelihoodtocost_elast)
    @test isnanorapprox(old_sens.theta_pointone.elasticity.pmp["A&C=f(A)"],  sens.theta_pointone.pmp.sens_costtolikelihood_elast)

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

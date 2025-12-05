using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
using Rasters, ArchGDAL
using OldConScape

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"

# The way the ascii wass read in is reversed and rotated from what GDAL does
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)[:, 1:58]
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)[:, 1:58]
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

isnanorapprox(xs::AbstractArray, ys::AbstractArray; atol=0.0) = 
    all(((x, y),) -> isnanorapprox(x, y; atol), zip(xs, ys))
function isnanorapprox(x::Number, y::Number; atol=0.0)
    out = (isnan(x) && isnan(y)) || isapprox(x, y; atol)
    out || println("Not approx: $x $y $atol")
    out
end

@testset "RSP sensitivity measure" begin
    # Set up the new sensitivity measures
    sensitivity_measures = (;
        # sum_sens_cost_sum =               SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=Summation()),
        # sum_sens_likelihood =             SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=Summation()),
        # sum_sens_costtolikelihood =       SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=Summation()),
        # sum_sens_likelihoodtocost =       SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=Summation()),
        # sum_sens_quality =                SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=Summation()),
        # sum_sens_source_quality =         SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=Summation()),
        # sum_sens_target_quality =         SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=Summation()),
        # sum_elast_cost =                  SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=Summation()),
        # sum_elast_likelihood =            SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=Summation()),
        # sum_elast_costtolikelihood =      SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=Summation()),
        # sum_elast_likelihoodtocost =      SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=Summation()),
        # sum_elast_quality =               SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=Summation()),
        # sum_elast_source_quality =        SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=Summation()),
        # sum_elast_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=Summation()),
        # eig_sens_cost =                   SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_likelihood =             SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_costtolikelihood =       SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_likelihoodtocost =       SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_quality =                SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_source_quality =         SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=EigMax()),
        # eig_sens_target_quality =         SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=EigMax()),
        # eig_elast_cost =                  SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=EigMax()),
        # eig_elast_likelihood =            SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=EigMax()),
        # eig_elast_costtolikelihood =      SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=EigMax()),
        # eig_elast_likelihoodtocost =      SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=EigMax()),
        # eig_elast_quality =               SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=EigMax()),
        # eig_elast_source_quality =        SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=EigMax()),
        eig_elast_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=EigMax()),
    )

    thetas = (theta_one=1.0, theta_pointone=0.1)
    grains = (grain_two=2, no_grain=nothing)
    proximity_measures = (ec=ExpectedCost(), pmp=PowerMeanProximity())
        
    problems = map(grains) do grain
        map(thetas) do theta
            map(proximity_measures) do proximity_measure
                movement= RandomisedShortestPath(; 
                    distance_transformation=ExpMinusAlpha(0.5),
                    proximity_measure, 
                    theta, 
                );
                ConScapeProblem(; movement, grain)
            end
        end
    end

    # OldConScape sensitivity
    @time old_sens = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
        g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
        cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
        metric = (elasticity=true, sensitivity=false)
        map(grains) do grain
            g_coarse = if isnothing(grain)
                g
            else
                OldConScape.Grid(size(steplikelihood)...;
                    affinities=affinities_sparse,
                    source_qualities=parent(quality),
                    target_qualities=OldConScape.coarse_graining(g, grain)
                )
            end
            map(thetas) do theta
                grsp = OldConScape.GridRSP(g_coarse; θ=theta)
                map(cfs) do connectivity_function
                    map(metric) do unitless
                        Dict(wrts .=> map(wrts) do wrt
                            # OldConScape cant do non-square Q
                            if !isnothing(grain) && wrt == "Q"
                                nothing
                            else
                                OldConScape.sensitivity(grsp;
                                    connectivity_function,
                                    distance_transformation=OldConScape.ExpMinus(),
                                    α=0.5,
                                    wrt,
                                    landscape_measure=["sum","eigenanalysis"][1],
                                    unitless,
                                    diagvalue=nothing,
                                    target_equal_source=true
                                )
                            end
                        end)
                    end
                end
            end
        end
    end;

    @time new_sens = map(problems) do problems_by_grain
        map(problems_by_grain) do problems_by_theta
            map(problems_by_theta) do problem
                solve(sensitivity_measures, problem, rast)
            end
        end
    end;

    @time old_x = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
        g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        grain = grains[2]
        connectivity_function = OldConScape.expected_cost
        wrt = "C"
        theta = thetas.theta_one
        metric = (elasticity=true, sensitivity=false)
        unitless = metric.sensitivity
        landscape_measures = ["sum","eigenanalysis"]
        landscape_measure = landscape_measures[2]

        g_coarse = if isnothing(grain)
            g
        else
            OldConScape.Grid(size(steplikelihood)...;
                affinities=affinities_sparse,
                source_qualities=parent(quality),
                target_qualities=OldConScape.coarse_graining(g, grain)
            )
        end
        grsp = OldConScape.GridRSP(g_coarse; θ=theta)
        if !isnothing(grain) && wrt == "Q"
            nothing
        else
            OldConScape.sensitivity(grsp;
                connectivity_function,
                distance_transformation=OldConScape.ExpMinus(),
                α=0.5,
                wrt,
                landscape_measure,
                unitless,
                diagvalue=nothing,
                target_equal_source=true
            )
        end
    end;
    @time new_x = solve(sensitivity_measures.sens_cost, problems[1].theta_one.ec, rast)
    heatmap(old_x)
    heatmap(new_x)
    old_x
    new_x

    isnanorapprox(old_x, new_x; atol=0.01)

    @assert OldConScape.store[].C == ConScape.store[].C
    @assert OldConScape.store[].W == ConScape.store[].W
    @assert OldConScape.store[].Z == ConScape.store[].Z
    @assert OldConScape.store[].Y == ConScape.store[].Y
    @assert OldConScape.store[].Zrows == ConScape.store[].Zrows
    @assert isapprox(OldConScape.store[].diag, ConScape.store[].diag; atol=1e-14)
    @assert isapprox(OldConScape.store[].diagC, ConScape.store[].diagC; atol=1e-14)
    OldConScape.store[].MᵀZ .- ConScape.store[].MᵀZ')
    using OldConScape.Plots
    minimum(OldConScape.store[].MᵀZ)
    minimum(ConScape.store[].MᵀZ')
    maximum(OldConScape.store[].MᵀZ)
    maximum(ConScape.store[].MᵀZ')

    isnanorapprox(OldConScape.store[].MᵀZ, ConScape.store[].MᵀZ'; atol=1e-1)
    @assert isapprox(OldConScape.store[].X5, ConScape.store[].X5'; atol=1e-10)
    OldConScape.store[].X5
    ConScape.store[].X5'
    @assert isapprox(OldConScape.store[].kB, ConScape.store[].kB)
    isnanorapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ; atol=1e-2)
    isnanorapprox(OldConScape.store[].kΣ.nzval, ConScape.store[].kΣ.nzval; atol=1e-2)
    findfirst(==(0.1661692401471833), OldConScape.store[].kΣ.nzval)
    OldConScape.store[].kΣ.nzval[2398]
    ConScape.store[].kΣ.nzval[2398]

    count(>(1e-15), OldConScape.store[].kΣ.nzval .- ConScape.store[].kΣ.nzval)
    sum(isapprox.(OldConScape.store[].kΣ.nzval, ConScape.store[].kΣ.nzval; atol=1e-2))

    old_sens[1].theta_one.ec.sensitivity["C"]
    new_sens[1].theta_one.ec.sens_cost
    @test isnanorapprox(old.sensitivity["C"],        new.sens_cost)

    heatmap(old_sens.grain_two.theta_pointone.ec.sensitivity["A"]; clims=(-3.0, 3.0))
    heatmap(new_sens.grain_two.theta_pointone.ec.sens_likelihood; clims=(-3.0, 3.0))
    heatmap(old_sens.grain_two.theta_pointone.ec.sensitivity["C"]; clims=(-2.0, 0.1))
    heatmap(new_sens.grain_two.theta_pointone.ec.sens_cost; clims=(-2.0, 0.1, ))
    
    sum(x -> isnan(x) ? 0.0 : x, old_sens.grain_two.theta_pointone.ec.sensitivity["A"])
    sum(x -> isnan(x) ? 0.0 : x, new_sens.grain_two.theta_pointone.ec.sens_likelihood)

    @testset "non-square, coarse graining of 2" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.no_grain, new_sens.no_grain, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                @test isnanorapprox(old.sensitivity["C"],        new.sens_cost; atol=1e-12)
                @test isnanorapprox(old.sensitivity["A"],        new.sens_likelihood; atol=1e-7)
                @test isnanorapprox(old.sensitivity["C&A=f(C)"], new.sens_likelihoodtocost; atol=1e-11)
                @test isnanorapprox(old.sensitivity["A&C=f(A)"], new.sens_costtolikelihood; atol=1e-7)
                @test isnanorapprox(old.sensitivity["Q"],        new.sens_quality; atol=1e-13)
                @test isnanorapprox(old.elasticity["C"],         new.elast_cost; atol=1e-11)
                @test isnanorapprox(old.elasticity["A"],         new.elast_likelihood; atol=1e-12)
                @test isnanorapprox(old.elasticity["C&A=f(C)"],  new.elast_likelihoodtocost; atol=1e-11)
                @test isnanorapprox(old.elasticity["A&C=f(A)"],  new.elast_costtolikelihood; atol=1e-11)
                @test isnanorapprox(old.elasticity["Q"],         new.elast_quality; atol=1e-13)
            end
        end
    end

    @testset "non-square, coarse graining of 2" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.grain_two, new_sens.grain_two, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                # OldConScape cant do non-square Q
                @test isnanorapprox(old.sensitivity["C"],        new.sens_cost; atol=1e-0)
                # @test isnanorapprox(old.sensitivity["A"],        new.sens_likelihood; atol=1)
                # @test isnanorapprox(old.sensitivity["C&A=f(C)"], new.sens_likelihoodtocost; atol=1)
                # @test isnanorapprox(old.sensitivity["A&C=f(A)"], new.sens_costtolikelihood; atol=1)
                @test isnanorapprox(old.elasticity["C"],         new.elast_cost; atol=1e-0)
                # @test isnanorapprox(old.elasticity["A"],         new.elast_likelihood; atol=1)
                # @test isnanorapprox(old.elasticity["C&A=f(C)"],  new.elast_likelihoodtocost; atol=1)
                # @test isnanorapprox(old.elasticity["A&C=f(A)"],  new.elast_costtolikelihood; atol=1)
            end
        end
    end

end

# include("sensitivity_simulation")

# @testset "RSP sensitivity measure" begin
#
#         test_g = OldConScape.Grid(size(steplikelihood)...;
#             affinities=affinities_sparse,
#             qualities=parent(quality),
#         )
#         wrts = ["A", "C", "Q", "C&A=f(C)", "A&C=f(A)"]
#         wrts = ["A", "C", "C&A=f(C)", "A&C=f(A)"]
#         cfs = (ec=OldConScape.expected_cost, pmp=OldConScape.power_mean_proximity)
#         connectivity_function = cfs.ec
#         wrt = "C"
#         theta = thetas.theta_pointone
#         unitless = false
#         map(thetas) do theta
#             sim_grsp = OldConScape.GridRSP(test_g; θ=theta)
#             map((elasticity=true, sensitivity=false)) do unitless
#                 map(cfs) do connectivity_function
#                     Dict(wrts .=> map(wrts) do wrt
#
#                     end)
#                 end
#             end
#         end
#
#         sensitivity_simulation(sim_grsp;
#             connectivity_function,
#             distance_transformation,
#             α,
#             wrt
#             landscape_measure
#             unitless::Bool=true,
#             target_equal_source,
#             one_out_of,
#         )

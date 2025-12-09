using ConScape, Test, SparseArrays, LinearAlgebra, Statistics
using Rasters, ArchGDAL
using OldConScape

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"

# The way the ascii wass read in is reversed and rotated from what GDAL does
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X) #[:, 1:58]
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X) #[:, 1:58]
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

isnanorapprox(xs::AbstractArray, ys::AbstractArray; atol=0.0) = 
    all(((x, y),) -> isnanorapprox(x, y; atol), zip(xs, ys))
function isnanorapprox(x::Number, y::Number; atol=0.0)
    out = (isnan(x) && isnan(y)) || isapprox(x, y; atol)
    out || println("Not approx: $x $y $atol")
    return out
end

@testset "RSP sensitivity measure" begin
    # Set up the new sensitivity measures
    sensitivity_measures = (;
        sum_sens_cost =                   SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=Summation()),
        sum_sens_likelihood =             SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=Summation()),
        sum_sens_costtolikelihood =       SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=Summation()),
        sum_sens_likelihoodtocost =       SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=Summation()),
        sum_sens_quality =                SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=Summation()),
        sum_sens_source_quality =         SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=Summation()),
        sum_sens_target_quality =         SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=Summation()),
        sum_elast_cost =                  SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=Summation()),
        sum_elast_likelihood =            SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=Summation()),
        sum_elast_costtolikelihood =      SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=Summation()),
        sum_elast_likelihoodtocost =      SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=Summation()),
        sum_elast_quality =               SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=Summation()),
        sum_elast_source_quality =        SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=Summation()),
        sum_elast_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=Summation()),
    )
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
        # eig_elast_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=EigMax()),
    # )

    thetas = (theta_one=1.0, theta_pointone=0.1)
    grains = (grain_two=2, no_grain=nothing)
    proximity_measures = (ec=ExpectedCost(), pmp=PowerMeanProximity())
        
    problems = map(grains) do grain
        map(thetas) do theta
            map(proximity_measures) do proximity_measure
                movement = RandomisedShortestPath(; 
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
            map(problems_by_theta) do proximity_problem
                solve(sensitivity_measures, proximity_problem, rast)
            end
        end
    end;

    OldConScape.store[] = (;)
    ConScape.store[] = (;)
    @time old_x = let
        affinities_sparse = OldConScape.graph_matrix_from_raster(parent(steplikelihood))
        g = OldConScape.Grid(size(steplikelihood)...;
            affinities=affinities_sparse,
            qualities=parent(quality),
        )
        grain = grains.grain_two
        connectivity_function = OldConScape.power_mean_proximity
        wrt = "C"
        theta = thetas.theta_pointone
        metric = (elasticity=true, sensitivity=false)
        unitless = metric.sensitivity
        landscape_measure = "sum"

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
    @time new_x = solve((; sens=sensitivity_measures.sum_sens_cost), problems.grain_two.theta_pointone.pmp, rast)
    cgi = init(problems.grain_two.theta_pointone.pmp, rast, 1)

    # @assert OldConScape.store[].C == ConScape.store[].C
    @assert OldConScape.store[].W == ConScape.store[].W
    @assert OldConScape.store[].Z == ConScape.store[].Z
    # @assert OldConScape.store[].Y == ConScape.store[].Y
    @assert OldConScape.store[].Zrows == ConScape.store[].Zrows'
    @assert OldConScape.store[].qᵗ == ConScape.store[].qᵗ
    @assert OldConScape.store[].qˢ == ConScape.store[].qˢ
    @assert isnanorapprox(OldConScape.store[].XdiagZⁱ, ConScape.store[].XdiagZⁱ)
    @assert isnanorapprox(OldConScape.store[].XZⁱ, ConScape.store[].XZⁱ)
    @assert isnanorapprox(OldConScape.store[].XᵀZ, ConScape.store[].XᵀZ')
    @assert isnanorapprox(OldConScape.store[].node_output, ConScape.store[].node_output[ConScape.sourceids(cgi)])
    @assert isnanorapprox(OldConScape.store[].edge_output, ConScape.store[].edge_output)
    

    @assert isnanorapprox(OldConScape.store[].K, ConScape.store[].K)
    @assert isnanorapprox(OldConScape.store[].M, ConScape.store[].M)
    @assert isnanorapprox(OldConScape.store[].diag, ConScape.store[].diag)
    @assert isnanorapprox(OldConScape.store[].diagC, ConScape.store[].diagC)
    @assert isnanorapprox(OldConScape.store[].MᵀZ, ConScape.store[].MᵀZ')
    @assert isnanorapprox(OldConScape.store[].C̄ᵣ, ConScape.store[].C̄ᵣ)
    @assert isnanorapprox(OldConScape.store[].X3, ConScape.store[].X3)
    @assert isnanorapprox(OldConScape.store[].X5, ConScape.store[].X5')
    @assert isnanorapprox(OldConScape.store[].kB, ConScape.store[].kB)
    @assert isnanorapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ)

    using OldConScape.Plots
    heatmap(old_x)
    heatmap(parent(new_x.sens))
    isnanorapprox(parent(new_x.sens), old_x)

    @testset "square Z, no coarse graining" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.no_grain, new_sens.no_grain, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                @test isnanorapprox(old.sensitivity["C"],        new.sum_sens_cost)
                @test isnanorapprox(old.sensitivity["A"],        new.sum_sens_likelihood)
                @test isnanorapprox(old.sensitivity["C&A=f(C)"], new.sum_sens_likelihoodtocost)
                @test isnanorapprox(old.sensitivity["A&C=f(A)"], new.sum_sens_costtolikelihood)
                @test isnanorapprox(old.elasticity["C"],         new.sum_elast_cost)
                @test isnanorapprox(old.elasticity["A"],         new.sum_elast_likelihood)
                @test isnanorapprox(old.elasticity["C&A=f(C)"],  new.sum_elast_likelihoodtocost)
                @test isnanorapprox(old.elasticity["A&C=f(A)"],  new.sum_elast_costtolikelihood)
                @test isnanorapprox(old.elasticity["Q"],         new.sum_elast_quality)
                @test isnanorapprox(old.sensitivity["Q"],        new.sum_sens_quality)
            end
        end
    end

    @testset "non-square, coarse graining of 2" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.grain_two, new_sens.grain_two, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                # OldConScape cant do non-square Q
                @test isnanorapprox(old.sensitivity["C"],        new.sum_sens_cost)
                @test isnanorapprox(old.sensitivity["A"],        new.sum_sens_likelihood)
                @test isnanorapprox(old.sensitivity["C&A=f(C)"], new.sum_sens_likelihoodtocost)
                @test isnanorapprox(old.sensitivity["A&C=f(A)"], new.sum_sens_costtolikelihood)
                @test isnanorapprox(old.elasticity["C"],         new.sum_elast_cost)
                @test isnanorapprox(old.elasticity["A"],         new.sum_elast_likelihood)
                @test isnanorapprox(old.elasticity["C&A=f(C)"],  new.sum_elast_likelihoodtocost)
                @test isnanorapprox(old.elasticity["A&C=f(A)"],  new.sum_elast_costtolikelihood)
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
#
#
    @time out = solve(sensitivity_measures, problems.no_grain.theta_one.ec, rast);
    @time out = solve(sensitivity_measures, problems.no_grain.theta_one.pmp, rast);
    @profview solve(sensitivity_measures, problems.no_grain.theta_one.ec, rast)
    @profview solve(sensitivity_measures, problems.no_grain.theta_one.pmp, rast)
    @descend solve(sensitivity_measures[1], problems.no_grain.theta_one.pmp, rast)



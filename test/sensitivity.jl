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

function isnanorapprox(xs::AbstractArray, ys::AbstractArray; 
    atol::Real=0,
    rtol::Real=Base.rtoldefault(LinearAlgebra.promote_leaf_eltypes(xs), LinearAlgebra.promote_leaf_eltypes(ys), atol),
)
    all(((x, y),) -> isnanorapprox(x, y; atol, rtol), zip(xs, ys))
end
function isnanorapprox(x::Number, y::Number; atol, rtol)
    out = (isnan(x) && isnan(y)) || isapprox(x, y; atol)
    out || println("Not approx: $x $y $atol")
    return out
end

@testset "RSP sensitivity measure" begin
    seed = 1234
    # Set up the new sensitivity measures
    sensitivity_measures = (;
        sens_sum_cost =                   SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=Summation()),
        sens_sum_likelihood =             SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=Summation()),
        sens_sum_costtolikelihood =       SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=Summation()),
        sens_sum_likelihoodtocost =       SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=Summation()),
        sens_sum_quality =                SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=Summation()),
        sens_sum_source_quality =         SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=Summation()),
        sens_sum_target_quality =         SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=Summation()),
        elast_sum_cost =                  SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=Summation()),
        elast_sum_likelihood =            SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=Summation()),
        elast_sum_costtolikelihood =      SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=Summation()),
        elast_sum_likelihoodtocost =      SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=Summation()),
        elast_sum_quality =               SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=Summation()),
        elast_sum_source_quality =        SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=Summation()),
        elast_sum_target_quality =        SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=Summation()),
        sens_eigen_cost =                 SensitivityAnalysis(; wrt=StepCost(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_likelihood =           SensitivityAnalysis(; wrt=StepLikelihood(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_costtolikelihood =     SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_likelihoodtocost =     SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_quality =              SensitivityAnalysis(; wrt=Quality(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_source_quality =       SensitivityAnalysis(; wrt=SourceQuality(), type=Sensitivity(), metric=EigMax(; seed)),
        sens_eigen_target_quality =       SensitivityAnalysis(; wrt=TargetQuality(), type=Sensitivity(), metric=EigMax(; seed)),
        elast_eigen_cost =                SensitivityAnalysis(; wrt=StepCost(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_likelihood =          SensitivityAnalysis(; wrt=StepLikelihood(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_costtolikelihood =    SensitivityAnalysis(; wrt=StepCostToLikelihood(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_likelihoodtocost =    SensitivityAnalysis(; wrt=StepLikelihoodToCost(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_quality =             SensitivityAnalysis(; wrt=Quality(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_source_quality =      SensitivityAnalysis(; wrt=SourceQuality(), type=Elasticity(), metric=EigMax(; seed)),
        elast_eigen_target_quality =      SensitivityAnalysis(; wrt=TargetQuality(), type=Elasticity(), metric=EigMax(; seed)),
    )

    thetas = (theta_one=1.0, theta_pointone=0.1)
    grains = (grain_two=2, no_grain=nothing)
    proximity_measures = (ec=ExpectedCost(), pmp=PowerMeanProximity())
        
    problems = map(grains) do grain
        map(thetas) do theta
            map(proximity_measures) do proximity_measure
                movement = RandomisedShortestPath(; 
                    distance_transformation=ExpMinusAlpha(1/2000),
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
        landscape_measures = (sum="sum", eigen="eigenanalysis")
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
                        map(landscape_measures) do landscape_measure
                            Dict(wrts .=> map(wrts) do wrt
                                # OldConScape cant do non-square Q
                                if !isnothing(grain) && wrt == "Q"
                                    nothing
                                else
                                    OldConScape.sensitivity(grsp;
                                        connectivity_function,
                                        distance_transformation=OldConScape.ExpMinus(),
                                        α=1/2000,
                                        wrt,
                                        landscape_measure,
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
        end
    end;

    @time new_sens = map(problems) do problems_by_grain
        map(problems_by_grain) do problems_by_theta
            map(problems_by_theta) do proximity_problem
                solve(sensitivity_measures, proximity_problem, rast)
            end
        end
    end;

    # @testset "old and new internals match" begin
    #     @test isapprox(OldConScape.store[].Fps[1].eigenvalues, ConScape.store[].Fps[1].eigenvalues)
    #     @test isapprox(OldConScape.store[].Fps[1].Q, ConScape.store[].Fps[1].Q)
    #     @test isapprox(OldConScape.store[].Fps[1].R, ConScape.store[].Fps[1].R)
    #     @test isapprox(OldConScape.store[].λ, ConScape.store[].λ)
    #     @test isnanorapprox(OldConScape.store[].targetnodes, ConScape.store[].targetnodes)
    #     @test isnanorapprox(OldConScape.store[].nontargetnodes, ConScape.store[].nontargetnodes)
    #     @test isnanorapprox(OldConScape.store[].Mtarget, ConScape.store[].Mtarget)
    #     @test isnanorapprox(OldConScape.store[].Mnontarget, ConScape.store[].Mnontarget)
    #     @test isnanorapprox(OldConScape.store[].MλI, ConScape.store[].MλI)
    #     @test isnanorapprox(OldConScape.store[].rhs, ConScape.store[].rhs)
    #     @test isnanorapprox(OldConScape.store[].vʳ₀, ConScape.store[].vʳ₀; atol=1e-12)
    #     @test isnanorapprox(OldConScape.store[].vʳ, ConScape.store[].vʳ; atol=1e-12)
    #     @test isnanorapprox(ConScape.store[].vˡ, ConScape.store[].v)
    #     @test isnanorapprox(ConScape.store[].vʳ, ConScape.store[].w)
    #     # Only the correlation is approximately equal, not the values
    #     @test isnanorapprox(cor(OldConScape.store[].vˡ, ConScape.store[].vˡ), 1)
    #     @test isapprox(maximum(OldConScape.store[].vˡ), maximum(ConScape.store[].vˡ))
    #     @test isapprox(maximum(OldConScape.store[].vˡ), 1)
    #     @test isnanorapprox(OldConScape.store[].MλI, ConScape.store[].MλI; atol=1e-13)
    #     @test isnanorapprox(OldConScape.store[].rhs, ConScape.store[].rhs; atol=1e-14)
    #     @test OldConScape.store[].C == ConScape.store[].C
    #     @test OldConScape.store[].W == ConScape.store[].W
    #     @test OldConScape.store[].Z == ConScape.store[].Z
    #     @test OldConScape.store[].Y == ConScape.store[].Y
    #     @test OldConScape.store[].Zrows == ConScape.store[].Zrows
    #     @test isapprox(OldConScape.store[].qˢ, ConScape.store[].qˢ)
    #     # qᵗ is only highly correlated, not actually approximate
    #     @test isapprox(cor(OldConScape.store[].qᵗ, ConScape.store[].qᵗ), 1)
    #     # But scaled identically
    #     @test isapprox(maximum(OldConScape.store[].qᵗ), maximum(ConScape.store[].qᵗ))
    #     @test isapprox(OldConScape.store[].K, ConScape.store[].K)
    #     @test isapprox(cor(vec(ConScape.store[].M), vec(OldConScape.store[].M)), 1)
    #     @test isapprox(OldConScape.store[].diagC, ConScape.store[].diagC)
    #     @test isapprox(OldConScape.store[].kB, ConScape.store[].kB)
    #     @test isapprox(OldConScape.store[].kΣ, ConScape.store[].kΣ)
    #     @test isapprox(OldConScape.store[].vTw, ConScape.store[].vTw)
    # end

    cgi = init(problems.grain_two.theta_one.ec, rast, 1)
    I = ConScape.sourceids(cgi)

    # old = old_sens.grain_two.theta_pointone.pmp
    # new = new_sens.grain_two.theta_pointone.pmp
    @testset "square Z, no coarse graining" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.no_grain, new_sens.no_grain, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                @testset "sum matches approximately" begin
                    @test isnanorapprox(old.sensitivity.sum["C"],          new.sens_sum_cost)
                    @test isnanorapprox(old.sensitivity.sum["A"],          new.sens_sum_likelihood)
                    @test isnanorapprox(old.sensitivity.sum["C&A=f(C)"],   new.sens_sum_likelihoodtocost)
                    @test isnanorapprox(old.sensitivity.sum["A&C=f(A)"],   new.sens_sum_costtolikelihood)
                    @test isnanorapprox(old.elasticity.sum["C"],           new.elast_sum_cost)
                    @test isnanorapprox(old.elasticity.sum["A"],           new.elast_sum_likelihood)
                    @test isnanorapprox(old.elasticity.sum["C&A=f(C)"],    new.elast_sum_likelihoodtocost)
                    @test isnanorapprox(old.elasticity.sum["A&C=f(A)"],    new.elast_sum_costtolikelihood)
                    @test isnanorapprox(old.elasticity.sum["Q"],           new.elast_sum_quality)
                    @test isnanorapprox(old.sensitivity.sum["Q"],          new.sens_sum_quality)
                end
                @testset "eigen is equally scaled and exactly correlated" begin
                    @test cor(old.sensitivity.eigen["C"][I],        new.sens_eigen_cost[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["A"][I],        new.sens_eigen_likelihood[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["C&A=f(C)"][I], new.sens_eigen_likelihoodtocost[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["A&C=f(A)"][I], new.sens_eigen_costtolikelihood[I]) ≈ 1
                    @test cor(old.elasticity.eigen["C"][I],         new.elast_eigen_cost[I]) ≈ 1
                    @test cor(old.elasticity.eigen["A"][I],         new.elast_eigen_likelihood[I]) ≈ 1
                    @test cor(old.elasticity.eigen["C&A=f(C)"][I],  new.elast_eigen_likelihoodtocost[I]) ≈ 1
                    @test cor(old.elasticity.eigen["A&C=f(A)"][I],  new.elast_eigen_costtolikelihood[I]) ≈ 1
                    @test maximum(abs, old.sensitivity.eigen["C"][I]) ≈        maximum(abs, new.sens_eigen_cost[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["A"][I]) ≈        maximum(abs,  new.sens_eigen_likelihood[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["C&A=f(C)"][I]) ≈ maximum(abs,  new.sens_eigen_likelihoodtocost[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["A&C=f(A)"][I]) ≈ maximum(abs,  new.sens_eigen_costtolikelihood[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["C"][I]) ≈         maximum(abs,  new.elast_eigen_cost[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["A"][I]) ≈         maximum(abs,  new.elast_eigen_likelihood[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["C&A=f(C)"][I]) ≈  maximum(abs,  new.elast_eigen_likelihoodtocost[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["A&C=f(A)"][I]) ≈  maximum(abs,  new.elast_eigen_costtolikelihood[I]) rtol=1e-5
                end
            end
        end
    end

    @testset "non-square, coarse graining of 2" begin
        @testset "theta $theta" for (old_by_theta, new_by_theta, theta) in zip(old_sens.grain_two, new_sens.grain_two, thetas)
            @testset "$pm" for (old, new, pm) in zip(old_by_theta, new_by_theta, proximity_measures)
                # OldConScape cant do non-square Q
                @testset "sum matches approximately" begin
                    @test isnanorapprox(old.sensitivity.sum["C"],          new.sens_sum_cost)
                    @test isnanorapprox(old.sensitivity.sum["A"],          new.sens_sum_likelihood)
                    @test isnanorapprox(old.sensitivity.sum["C&A=f(C)"],   new.sens_sum_likelihoodtocost)
                    @test isnanorapprox(old.sensitivity.sum["A&C=f(A)"],   new.sens_sum_costtolikelihood)
                    @test isnanorapprox(old.elasticity.sum["C"],           new.elast_sum_cost)
                    @test isnanorapprox(old.elasticity.sum["A"],           new.elast_sum_likelihood)
                    @test isnanorapprox(old.elasticity.sum["C&A=f(C)"],    new.elast_sum_likelihoodtocost)
                    @test isnanorapprox(old.elasticity.sum["A&C=f(A)"],    new.elast_sum_costtolikelihood)
                end
                @testset "eigen is equally scaled and exactly correlated" begin
                    @test cor(old.sensitivity.eigen["C"][I],        new.sens_eigen_cost[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["A"][I],        new.sens_eigen_likelihood[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["C&A=f(C)"][I], new.sens_eigen_likelihoodtocost[I]) ≈ 1
                    @test cor(old.sensitivity.eigen["A&C=f(A)"][I], new.sens_eigen_costtolikelihood[I]) ≈ 1
                    @test cor(old.elasticity.eigen["C"][I],         new.elast_eigen_cost[I]) ≈ 1
                    @test cor(old.elasticity.eigen["A"][I],         new.elast_eigen_likelihood[I]) ≈ 1
                    @test cor(old.elasticity.eigen["C&A=f(C)"][I],  new.elast_eigen_likelihoodtocost[I]) ≈ 1
                    @test cor(old.elasticity.eigen["A&C=f(A)"][I],  new.elast_eigen_costtolikelihood[I]) ≈ 1
                    @test maximum(abs, old.sensitivity.eigen["C"][I]) ≈        maximum(abs, new.sens_eigen_cost[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["A"][I]) ≈        maximum(abs,  new.sens_eigen_likelihood[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["C&A=f(C)"][I]) ≈ maximum(abs,  new.sens_eigen_likelihoodtocost[I]) rtol=1e-5
                    @test maximum(abs, old.sensitivity.eigen["A&C=f(A)"][I]) ≈ maximum(abs,  new.sens_eigen_costtolikelihood[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["C"][I]) ≈         maximum(abs,  new.elast_eigen_cost[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["A"][I]) ≈         maximum(abs,  new.elast_eigen_likelihood[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["C&A=f(C)"][I]) ≈  maximum(abs,  new.elast_eigen_likelihoodtocost[I]) rtol=1e-5
                    @test maximum(abs, old.elasticity.eigen["A&C=f(A)"][I]) ≈  maximum(abs,  new.elast_eigen_costtolikelihood[I]) rtol=1e-5
                end
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

    function f(cgi, n)  
        for i in 1:n
            ti = init(cgi, i)
            ConScape._proximitymatrixcol(ti)
        end
    end
    @descend f(, 300)
    cgi = init(problems.no_grain.theta_one.ec, rast, 1)
    ti = init(cgi, 1)
    pm = ConScape.proximity_measure(ti)
    @inferred ConScape._proximitymatrixcol(ti)
    @descend ConScape.get_or_compute_target!(ti, pm)
    @time out = solve(sensitivity_measures, problems.no_grain.theta_one.ec, rast);
    @time out = solve(sensitivity_measures, problems.no_grain.theta_one.pmp, rast);
    @profview solve(sensitivity_measures, problems.no_grain.theta_one.ec, rast)
    @profview solve(sensitivity_measures, problems.no_grain.theta_one.pmp, rast)
    @descend solve(sensitivity_measures[1], problems.no_grain.theta_one.pmp, rast)



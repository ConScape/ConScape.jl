using ConScape, Test, SparseArrays, LinearAlgebra, Statistics

include("simulation.jl")

function validate_sensitivity(analytical::AbstractMatrix{Float64}, simulated::AbstractMatrix{Float64})::Tuple{Float64,Float64}
    valid_idx = .!isnan.(analytical) .& .!isnan.(simulated)
    any(valid_idx) || return (NaN, NaN)
    a, s = analytical[valid_idx], simulated[valid_idx]
    nonzero = abs.(a) .> 1e-10
    mean_err = any(nonzero) ? mean(abs.(a[nonzero] .- s[nonzero]) ./ abs.(a[nonzero])) : 0.0
    return cor(a, s), mean_err
end

# 10x20 8-connected grid with quality gradient (ensures non-trivial quality sensitivities).
# No walls: finite-difference requires small perturbations, but walls have ~1e-20 likelihoods
# where adding ε=1e-6 is a 10^14x change, not a small perturbation.
# Wall cases are validated against OldConScape in comparison.jl.
nrows, ncols = 10, 20
quality = Matrix{Float64}(reshape(collect(200:-1:1), ncols, nrows) |> permutedims)
steplikelihood = ConScape._generate_likelihood(nrows, ncols, 8) .* 0.5
g = ConScape.GridGraph(; steplikelihood, quality)
θ = 0.5

types = (Sensitivity(), Elasticity())

# RSP supports all sensitivity types
# Note: PowerMeanProximity excluded due to Julia 1.12 compiler segfault during type inference
rsp_wrts = (StepLikelihood(), StepCost(), Quality(), StepCostToLikelihood(), StepLikelihoodToCost())
for wrt in rsp_wrts
    for type in types
        movement = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
        measures = (; sens=SensitivityAnalysis(; wrt, metric=Summation(), type))
        problem = ConScapeProblem(; movement, measures)
        analytical = solve(problem, g).sens
        simulated = sensitivity_simulation(problem, g; wrt, metric=Summation(), type)
        corr, mean_err = validate_sensitivity(analytical, simulated)
        @test corr > 0.999
        @test mean_err < 0.01
    end
end

# RandomWalk only supports Quality sensitivity (not Permeability types)
rw_wrts = (Quality(),)
for wrt in rw_wrts
    for type in types
        movement = RandomWalk()
        measures = (; sens=SensitivityAnalysis(; wrt, metric=Summation(), type))
        problem = ConScapeProblem(; movement, measures)
        analytical = solve(problem, g).sens
        simulated = sensitivity_simulation(problem, g; wrt, metric=Summation(), type)
        corr, mean_err = validate_sensitivity(analytical, simulated)
        @test corr > 0.999
        @test mean_err < 0.01
    end
end

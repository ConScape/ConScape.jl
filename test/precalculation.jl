using ConScape, Test, SparseArrays, LinearAlgebra, Graphs, SimpleWeightedGraphs
using Rasters, ArchGDAL

# Load test data
datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
landscape = "sno_2000"
steplikelihood = reverse(rotr90(Raster(joinpath(datadir, "affinities_$landscape.asc"); missingval=NaN)); dims=X)
quality = reverse(rotr90(Raster(joinpath(datadir, "qualities_$landscape.asc"); missingval=NaN)); dims=X)
quality[(steplikelihood .> 0) .& isnan.(quality)] .= 1e-20
rast = RasterStack((; steplikelihood, quality))

θ = 0.1

@testset "sparse_precalculation" begin
    @testset "RSP" begin
        rsp = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=rsp, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cg = ConScape.connectedgraphs(ggi)[1]

        precalc = ConScape.sparse_precalculation(problem, cg)

        # Check all expected keys are present
        @test haskey(precalc, :P)
        @test haskey(precalc, :W)
        @test haskey(precalc, :IW)
        @test haskey(precalc, :IW_adj)
        @test haskey(precalc, :C)
        @test haskey(precalc, :CW)
        @test haskey(precalc, :CW_t)
        @test haskey(precalc, :F_IW)
        @test haskey(precalc, :F_IW_adj)
        @test haskey(precalc, :A)
        @test haskey(precalc, :Aⁱ)
        @test haskey(precalc, :A_rowsums)
        @test haskey(precalc, :θ)
        @test haskey(precalc, :qᵗ)
        @test haskey(precalc, :qˢ)

        # Check types
        @test precalc.P isa SparseMatrixCSC{Float64,Int64}
        @test precalc.W isa SparseMatrixCSC{Float64,Int64}
        @test precalc.IW isa SparseMatrixCSC{Float64,Int64}
        @test precalc.IW_adj isa Union{Adjoint{Float64,<:SparseMatrixCSC},SparseMatrixCSC{Float64,Int64}}
        @test precalc.C isa SparseMatrixCSC{Float64,Int64}
        @test precalc.CW isa SparseMatrixCSC{Float64,Int64}
        @test precalc.CW_t isa SparseMatrixCSC{Float64,Int64}
        @test precalc.F_IW isa Factorization
        @test precalc.F_IW_adj isa Union{Adjoint,Factorization}
        @test precalc.A isa SparseMatrixCSC{Float64,Int64}
        @test precalc.Aⁱ isa SparseMatrixCSC{Float64,Int64}
        @test precalc.A_rowsums isa ConScape.ReadOnlyArray
        @test precalc.θ == θ
        @test precalc.qᵗ isa ConScape.ReadOnlyArray
        @test precalc.qˢ isa ConScape.ReadOnlyArray

        # Check P is row-stochastic (rows sum to 1)
        P_rowsums = sum(precalc.P, dims=2)
        @test all(isapprox.(P_rowsums, 1.0, atol=1e-10))

        # Check W is substochastic (rows sum to <= 1)
        W_rowsums = sum(precalc.W, dims=2)
        @test all(W_rowsums .<= 1.0 + 1e-10)

        # Check IW = I - W
        @test precalc.IW ≈ I - precalc.W

        # Check CW = C .* W
        @test precalc.CW ≈ precalc.C .* precalc.W

        # Check CW_t is transpose of CW
        @test precalc.CW_t ≈ transpose(precalc.CW)
    end

    @testset "LCP" begin
        lcp = LCP(; distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=lcp, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cg = ConScape.connectedgraphs(ggi)[1]

        precalc = ConScape.sparse_precalculation(problem, cg)

        # Check all expected keys are present
        @test haskey(precalc, :P)
        @test haskey(precalc, :L_rowsums)
        @test haskey(precalc, :cost_weighted_digraph)
        @test haskey(precalc, :path_allocs)
        @test haskey(precalc, :qᵗ)
        @test haskey(precalc, :qˢ)

        # Check types
        @test precalc.P isa SparseMatrixCSC{Float64,Int64}
        @test precalc.L_rowsums isa ConScape.ReadOnlyArray
        @test precalc.cost_weighted_digraph isa SimpleWeightedDiGraph
        @test precalc.path_allocs isa Vector{Vector{Int64}}
        @test precalc.qᵗ isa ConScape.ReadOnlyArray
        @test precalc.qˢ isa ConScape.ReadOnlyArray

        # Check P is row-stochastic
        P_rowsums = sum(precalc.P, dims=2)
        @test all(isapprox.(P_rowsums, 1.0, atol=1e-10))
    end

    @testset "RandomWalk" begin
        rw = RandomWalk(; distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=rw, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cg = ConScape.connectedgraphs(ggi)[1]

        precalc = ConScape.sparse_precalculation(problem, cg)

        # Check all expected keys are present
        @test haskey(precalc, :Lⁱ)
        @test haskey(precalc, :L_rowsums)
        @test haskey(precalc, :P)
        @test haskey(precalc, :PC)
        @test haskey(precalc, :PC_rowsums)
        @test haskey(precalc, :IP)
        @test haskey(precalc, :F_IP)
        @test haskey(precalc, :qᵗ)
        @test haskey(precalc, :qˢ)

        # Check types
        @test precalc.Lⁱ isa SparseMatrixCSC{Float64,Int64}
        @test precalc.L_rowsums isa ConScape.ReadOnlyArray
        @test precalc.P isa SparseMatrixCSC{Float64,Int64}
        @test precalc.PC isa SparseMatrixCSC{Float64,Int64}
        @test precalc.PC_rowsums isa Matrix{Float64}
        @test precalc.IP isa SparseMatrixCSC{Float64,Int64}
        @test precalc.F_IP isa Factorization
        @test precalc.qᵗ isa ConScape.ReadOnlyArray
        @test precalc.qˢ isa ConScape.ReadOnlyArray

        # Check P is row-stochastic
        P_rowsums = sum(precalc.P, dims=2)
        @test all(isapprox.(P_rowsums, 1.0, atol=1e-10))

        # Check IP = I - P
        @test precalc.IP ≈ I - precalc.P

        # Check PC = P .* C
        C = ConScape.stepcost(cg)
        @test precalc.PC ≈ precalc.P .* C
    end

    @testset "Euclidean" begin
        euc = Euclidean()
        problem = ConScapeProblem(; movement=euc, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cg = ConScape.connectedgraphs(ggi)[1]

        precalc = ConScape.sparse_precalculation(problem, cg)

        # Euclidean should return empty NamedTuple
        @test precalc == (;)
    end
end

@testset "dense_precalculation" begin
    @testset "RSP without full matrix measures" begin
        # FunctionalHabitat doesn't need full matrices
        rsp = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=rsp, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cgi = init(ggi, 1)

        dense_precalc = ConScape.dense_precalculation(cgi)
        @test dense_precalc == (;)
    end

    @testset "RSP with all dense precalculations" begin
        rsp = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
        # Measures that trigger all dense precalculation paths:
        # - EdgeBetweenness: Z_full, Zrows_full
        # - EigMax: Z_full, eigmax
        # - SensitivityAnalysis{StepLikelihood} with Summation: sum_sensitivity_precursors
        # - SensitivityAnalysis{StepLikelihood} with EigMax: eigmax_sensitivity_precursors
        # - SensitivityAnalysis with ExpectedCost proximity: Y_full
        measures = (;
            eb = EdgeBetweenness(QualityWeighted()),
            eigmax = EigMax(),
            sens_sum = SensitivityAnalysis(; wrt=StepLikelihood(), metric=Summation()),
            sens_eigmax = SensitivityAnalysis(; wrt=StepLikelihood(), metric=EigMax()),
        )
        problem = ConScapeProblem(; movement=rsp, measures)
        ggi = init(problem, rast)
        cgi = init(ggi, 1)

        # Check needs_ traits
        @test ConScape.needs_full_fundamentalmatrix(measures.eb, rsp) == true
        @test ConScape.needs_full_fundamentalrowmatrix(measures.eb, rsp) == true
        @test ConScape.needs_full_fundamentalmatrix(measures.sens_sum, rsp) == true
        @test ConScape.needs_full_fundamentalrowmatrix(measures.sens_sum, rsp) == true
        @test ConScape.needs_full_costdistancematrix(measures.sens_sum, rsp) == true
        @test ConScape.needs_eigmax(measures.eigmax, rsp) == true
        @test ConScape.needs_eigmax(measures.sens_eigmax, rsp) == true
        @test ConScape.needs_sum_sensitivity_precursors(measures.sens_sum, rsp) == true
        @test ConScape.needs_eigmax_sensitivity_precursors(measures.sens_eigmax, rsp) == true

        # dense_precalculation is already called during init(), so we access the precalculation
        dense_precalc = ConScape.precalculation(cgi)

        # Check all expected keys are present
        @test haskey(dense_precalc, :Z_full)
        @test haskey(dense_precalc, :Zrows_full)
        @test haskey(dense_precalc, :Y_full)
        @test haskey(dense_precalc, :eigmax)
        @test haskey(dense_precalc, :sum_sensitivity_precursors)
        @test haskey(dense_precalc, :eigmax_sensitivity_precursors)

        # Check types
        @test dense_precalc.Z_full isa Matrix{Float64}
        @test dense_precalc.Zrows_full isa Matrix{Float64}
        @test dense_precalc.Y_full isa Matrix{Float64}
        @test dense_precalc.eigmax isa Tuple{Vector{Float64},Float64,Vector{Float64}}
        @test dense_precalc.sum_sensitivity_precursors isa NamedTuple
        @test dense_precalc.eigmax_sensitivity_precursors isa NamedTuple

        # Check matrix dimensions
        n = ConScape.nsources(cgi)
        m = ConScape.ntargets(cgi)
        @test size(dense_precalc.Z_full) == (n, m)
        @test size(dense_precalc.Zrows_full) == (m, n)
        @test size(dense_precalc.Y_full) == (n, m)

        # Check eigmax eigenvalue is positive
        (vˡ, λ, vʳ) = dense_precalc.eigmax
        @test λ > 0
        @test length(vˡ) == n
        @test length(vʳ) == n
    end

    @testset "RandomWalk without full matrix measures" begin
        rw = RandomWalk(; distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=rw, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cgi = init(ggi, 1)

        # EigMax is only defined for RSP, not RandomWalk
        @test ConScape.needs_eigmax(EigMax(), rw) == false

        dense_precalc = ConScape.dense_precalculation(cgi)
        @test dense_precalc == (;)
    end

    @testset "LCP dense_precalculation" begin
        lcp = LCP(; distance_transformation=ExpMinus())
        problem = ConScapeProblem(; movement=lcp, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cgi = init(ggi, 1)

        # LCP should return empty for dense precalculation
        dense_precalc = ConScape.dense_precalculation(cgi)
        @test dense_precalc == (;)
    end

    @testset "Euclidean dense_precalculation" begin
        euc = Euclidean()
        problem = ConScapeProblem(; movement=euc, measures=(; fh=FunctionalHabitat()))
        ggi = init(problem, rast)
        cgi = init(ggi, 1)

        # Euclidean should return empty for dense precalculation
        dense_precalc = ConScape.dense_precalculation(cgi)
        @test dense_precalc == (;)
    end
end

@testset "target_precalculation! RSP" begin
    rsp = RSP(ExpectedCost(); theta=θ, distance_transformation=ExpMinus())
    problem = ConScapeProblem(; movement=rsp, measures=(; fh=FunctionalHabitat()))
    ggi = init(problem, rast)
    cgi = init(ggi, 1)
    ti = init(cgi, 1)

    @testset "Z (fundamental matrix column)" begin
        Z = ti.Z
        @test Z isa ConScape.ReadOnlyArray{Float64,1}
        @test length(Z) == ConScape.nsources(cgi)
        @test all(isfinite.(Z))
        @test all(Z .> 0)  # Fundamental matrix entries should be positive
    end

    @testset "Zⁱ (inverse fundamental matrix column)" begin
        ti2 = init(cgi, 1)  # Fresh target init
        Zⁱ = ti2.Zⁱ
        @test Zⁱ isa ConScape.ReadOnlyArray{Float64,1}
        @test length(Zⁱ) == ConScape.nsources(cgi)
        # Zⁱ should be approximately 1/Z where Z is not too small
        Z = ti2.Z
        for i in eachindex(Z)
            if Z[i] > 1e-10
                @test isapprox(Zⁱ[i], 1/Z[i], rtol=1e-6)
            end
        end
    end

    @testset "Q (quality matrix column)" begin
        ti3 = init(cgi, 1)
        Q = ti3.Q
        @test Q isa ConScape.ReadOnlyArray{Float64,1}
        @test length(Q) == ConScape.nsources(cgi)
        # Q = qˢ .* qᵗ
        @test Q ≈ ti3.qˢ .* ti3.qᵗ
    end

    @testset "K (proximity matrix column)" begin
        ti4 = init(cgi, 1)
        K = ti4.K
        @test K isa ConScape.ReadOnlyArray{Float64,1}
        @test length(K) == ConScape.nsources(cgi)
        @test all(K .>= 0)  # Proximity should be non-negative
    end

    @testset "M (landscape matrix column)" begin
        ti5 = init(cgi, 1)
        M = ti5.M
        @test M isa ConScape.ReadOnlyArray{Float64,1}
        @test length(M) == ConScape.nsources(cgi)
        # M = qˢ .* K .* qᵗ
        @test M ≈ ti5.qˢ .* ti5.K .* ti5.qᵗ
    end
end

@testset "target_precalculation! LCP" begin
    lcp = LCP(; distance_transformation=ExpMinus())
    problem = ConScapeProblem(; movement=lcp, measures=(; fh=FunctionalHabitat()))
    ggi = init(problem, rast)
    cgi = init(ggi, 1)
    ti = init(cgi, 1)

    @testset "shortest_paths" begin
        sp = ti.shortest_paths
        @test sp isa Graphs.DijkstraState
        @test length(sp.dists) == ConScape.nsources(cgi)
        @test sp.dists[ConScape.targetnode(ti)] == 0.0  # Distance to self is 0
    end

    @testset "K (proximity from shortest paths)" begin
        ti2 = init(cgi, 1)
        K = ti2.K
        @test K isa ConScape.ReadOnlyArray{Float64,1}
        @test length(K) == ConScape.nsources(cgi)
        @test all(K .>= 0)
        @test K[ConScape.targetnode(ti2)] == 1.0  # exp(-0) = 1
    end

    @testset "Q (quality matrix column)" begin
        ti3 = init(cgi, 1)
        Q = ti3.Q
        @test Q isa ConScape.ReadOnlyArray{Float64,1}
        @test Q ≈ ti3.qˢ .* ti3.qᵗ
    end

    @testset "M (landscape matrix column)" begin
        ti4 = init(cgi, 1)
        M = ti4.M
        @test M isa ConScape.ReadOnlyArray{Float64,1}
        @test M ≈ ti4.qˢ .* ti4.K .* ti4.qᵗ
    end
end

@testset "target_precalculation! RandomWalk" begin
    rw = RandomWalk(; distance_transformation=ExpMinus())
    problem = ConScapeProblem(; movement=rw, measures=(; fh=FunctionalHabitat()))
    ggi = init(problem, rast)
    cgi = init(ggi, 1)
    ti = init(cgi, 1)

    @testset "W (modified transition matrix)" begin
        W = ti.W
        @test W isa SparseMatrixCSC{Float64,Int64}
        # Target row should be zeroed out
        target_row = W[ConScape.targetnode(ti), :]
        @test all(target_row .== 0)
    end

    @testset "Z (fundamental matrix column)" begin
        ti2 = init(cgi, 1)
        Z = ti2.Z
        @test Z isa ConScape.ReadOnlyArray{Float64,1}
        @test length(Z) == ConScape.nsources(cgi)
    end

    @testset "Q (quality matrix column)" begin
        ti3 = init(cgi, 1)
        Q = ti3.Q
        @test Q isa ConScape.ReadOnlyArray{Float64,1}
        @test Q ≈ ti3.qˢ .* ti3.qᵗ
    end

    @testset "K (proximity matrix column)" begin
        ti4 = init(cgi, 1)
        K = ti4.K
        @test K isa ConScape.ReadOnlyArray{Float64,1}
        @test all(K .>= 0)
    end

    @testset "M (landscape matrix column)" begin
        ti5 = init(cgi, 1)
        M = ti5.M
        @test M isa ConScape.ReadOnlyArray{Float64,1}
        @test M ≈ ti5.qˢ .* ti5.K .* ti5.qᵗ
    end
end

@testset "helper functions" begin
    @testset "_transitionprobability" begin
        L = sprand(10, 10, 0.3)
        L = L + L'  # Make symmetric
        for i in 1:10
            L[i, i] = 0  # No self-loops
        end
        dropzeros!(L)

        P, rowsums = ConScape._transitionprobability(L)

        @test P isa SparseMatrixCSC
        @test rowsums isa ConScape.ReadOnlyArray

        # P should be row-stochastic
        P_rowsums = sum(P, dims=2)
        @test all(isapprox.(P_rowsums, 1.0, atol=1e-10))

        # P[i,j] = L[i,j] / sum(L[i,:])
        for i in 1:10
            if rowsums[i] > 0
                for j in 1:10
                    if L[i,j] > 0
                        @test isapprox(P[i,j], L[i,j] / rowsums[i], atol=1e-10)
                    end
                end
            end
        end
    end

    @testset "_substochasticmatrix" begin
        L = sprand(10, 10, 0.3)
        L = L + L'
        for i in 1:10
            L[i, i] = 0
        end
        dropzeros!(L)

        P, _ = ConScape._transitionprobability(L)
        C = sprand(10, 10, 0.3)
        C = C + C'
        dropzeros!(C)

        rsp = RSP(; theta=0.5)
        W = ConScape._substochasticmatrix(rsp, P, C)

        @test W isa SparseMatrixCSC

        # W should be substochastic (row sums <= 1)
        W_rowsums = sum(W, dims=2)
        @test all(W_rowsums .<= 1.0 + 1e-10)

        # W = P .* exp(-θ * C)
        expected_W = P .* exp.(-0.5 .* C)
        @test W ≈ expected_W
    end

    @testset "_inv" begin
        # Test custom inv that avoids Inf
        @test ConScape._inv(2.0) == 0.5
        @test ConScape._inv(0.0) == floatmax(Float64)
        @test isfinite(ConScape._inv(1e-400))
    end

    @testset "_identity_col!" begin
        ws = zeros(10)
        target = ConScape.TargetID(CartesianIndex(1,1), 1, 1, 5)
        result = ConScape._identity_col!(ws, target)

        @test result[5] == 1.0
        @test sum(result) == 1.0
        @test count(x -> x == 0.0, result) == 9
    end
end

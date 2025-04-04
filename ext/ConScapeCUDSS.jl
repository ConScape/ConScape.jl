module ConScapeCUDSS

using ConScape
using CUDA
using CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra

using ConScape: FundamentalMeasure, AbstractProblem, Grid, setup_sparse_problem

struct CUDSSsolver <: Solver end

function ConScape.solve(m::CUDSSsolver, cm::FundamentalMeasure, p::AbstractProblem, g::Grid) 
    (; A, B, Pref, W) = setup_sparse_problem(g, cm)
    Z = zeros(T, size(B))

    A_gpu = CuSparseMatrixCSR(A |> tril)
    Z_gpu = CuMatrix(Z)
    B_gpu = CuMatrix(B)

    solver = CudssSolver(A_gpu, "S", "L")

    cudss("analysis", solver, Z_gpu, B_gpu)
    cudss("factorization", solver, Z_gpu, B_gpu)
    cudss("solve", solver, Z_gpu, B_gpu)

    Z .= Z_gpu
    # TODO: maybe graph measures can run on GPU as well?
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    results = map(p.graph_measures) do gm
        compute(gm, p, grsp)
    end 
    return _merge_to_stack(results)
end

end
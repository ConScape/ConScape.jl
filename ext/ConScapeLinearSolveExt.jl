module ConScapeLinearSolveExt

using ConScape
using LinearSolve
import CommonSolve
import LinearAlgebra

# Now LinearSolve.jl is imported, add a working LinearSolver constructor
ConScape.LinearSolver(args::A, kw::K) where {A<:Tuple,K} =
    LinearSolver{A,K}(args, kw)

function CommonSolve.init(solver::LinearSolver, A::AbstractMatrix)
    b = ones(eltype(A), size(A, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, solver.args...; solver.keywords...)
    return linsolve
end

function LinearAlgebra.ldiv!(s::LinearSolver, B, init, B_copy)
    linsolve = init
    reinit!(linsolve; b=vec(B_copy), reuse_precs=true)
    sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
    vec(B) .= sol.u
    return B
end

end

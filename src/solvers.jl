"""
   VectorSolver(; check, threaded)

Use Julias' default UMFPACK solver, 
over vector columns for each target of the problem.
"""
struct VectorSolver <: Solver end

# Solver init
init(solver::VectorSolver, A::AbstractMatrix) = lu(A)

# ldiv!
# The main reason to have Solver types is to provide methods for ldiv!
function LinearAlgebra.ldiv!(p::Initialisation, init, B)
    # Handle using a workspace instead of copying B
    B_copy = take!(workspaces(p)) .= B
    # Solve
    X = ldiv!(solver(p), B, init, B_copy)
    # Return the workspace to the pool
    put!(workspaces(p), B_copy)
    return X
end
LinearAlgebra.ldiv!(s::VectorSolver, B, F, B_copy) = ldiv!(B, F, B_copy)

"""
   LinearSolver(args...; kw...)

Solve all operations column-by-column using LinearSolve.jl solvers.

Arguments and keywords are passed to `LinearSolve.solve` after the
problem object, like:

````julia
`LinearSolve.solve(linearproblem, args...; kw...)`
````

# Example

This example uses LinearSolve.jl wth `KrylovJL_GMRES` and a preconditioner.

TODO: an example that is realistic

````julia
using LinearSolve
measures = (;
    func=ConnectedHabitat(),
    qbetw=Betweenness(QualityWeighted()),
)
movement = RandomisedShortestPath(ExpectedCost(); theta=1.0)
solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
problem = ConScape.Problem(measures; movement, solver) 

rast = RasterStack((source_qualities="source_qs.tif", target_qualities="target_qs.tif"))
result = solve(problem, rast)
````
"""
struct LinearSolver{A<:Tuple,K} <: Solver
    args::A
    keywords::K
    # Constructors error without LinearSolve.jl loaded
    function LinearSolver(args, kw) 
        args isa Tuple || throw(ArgumentError("args must be a Tuple"))
        error("First run `using LinearSolve` to use LinearSolver")
    end
    LinearSolver{A,K}(args::A, kw::K) where {A,K} = new{A,K}(args, kw)
end
LinearSolver(args...; kw...) = LinearSolver(args, kw)

# See method implementations in ext/ConScapeLinearSolveExt.jl

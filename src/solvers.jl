"""
   VectorSolver(; check, threaded)

Use Julias' default UMFPACK solver, 
over vector columns for each target of the problem.
"""
@kwdef struct VectorSolver <: Solver
    check::Bool = true
    threaded::Bool = false
end

"""
   LinearSolver(args...; threaded, kw...)

Solve all operations column-by-column using LinearSolve.jl solvers.

The `threaded` keyword specifies if threads are used per target.
Other arguments and keywords are passed to `LinearSolve.solve` after the
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
    threaded::Bool
    # Constructors error without LinearSolve.jl loaded
    function LinearSolver(args, kw, threaded) 
        args isa Tuple || throw(ArgumentError("args must be a Tuple"))
        threaded isa Bool || throw(ArgumentError("threaded must be a Bool"))
        error("First run `using LinearSolve` to use LinearSolver")
    end
    LinearSolver{A,K}(args::A, kw::K, threaded::Bool) where {A,K} =
        new{A,K}(args, kw, threaded)
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

# See implementation in ext/ConScapeLinearSolveExt.jl

isthreaded(s::Solver) = false
isthreaded(s::LinearSolver) = s.threaded
isthreaded(s::VectorSolver) = s.threaded

# Solver init
function init(solver::VectorSolver, A::AbstractMatrix)
    F = lu(A)
    if isthreaded(solver)
        nbuffers = Threads.nthreads()
        channel = Channel{typeof(F)}(nbuffers)
        for _ in 1:nbuffers
            put!(channel, copy(F))
        end
        return channel
    else
        return F
    end
end

# ldiv!
# The main reason to have solvers is to provide methods for ldiv!
function LinearAlgebra.ldiv!(p::Initialisation, init, B)
    # Handle using a workspace instead of copying B
    B_copy = take!(workspaces(p)) .= B
    # Solve
    X = ldiv!(solver(p), B, init, B_copy)
    # Return the workspace to the pool
    put!(workspaces(p), B_copy)
    return X
end
function LinearAlgebra.ldiv!(s::VectorSolver, B, init, B_copy)
    if isthreaded(s)
        channel = init
        F = take!(channel)
        # Solve for the column
        ldiv!(B, F, B_copy)
        # Reuse the workspace 
        put!(channel, F)
    else
        F = init
        ldiv!(B, F, B_copy)
    end
    return B
end
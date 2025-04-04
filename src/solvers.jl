"""
   VectorSolver(; check, threaded)

Use julias default solver over vector colums of the problem.
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
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor()),
problem = ConScape.Problem(; 
    solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
    measures = (;
        func=ConnectedHabitat(),
        qbetw=Betweenness(QualityWeighted()),
    ),
    movement_mode = RandomisedShortestPath(ExpectedCost(), theta=1.0),
)
````
"""
struct LinearSolver{A,K} <: Solver
    args::A
    keywords::K
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

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
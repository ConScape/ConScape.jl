# Recusive getters for nested problems
measures(p::AbstractProblem) = measures(problem(p))
movement_mode(p::AbstractProblem) = movement_mode(problem(p))
proximity_measure(p::AbstractProblem) = proximity_measure(problem(p))
connectivity_function(p::AbstractProblem) =
    connectivity_function(proximity_measure(p))
isthreaded(p::AbstractProblem) = false

"""
    Problem(measures; movement_mode, solver, θ)

A `Problem` specifies a `NamedTuple of `Measure`s to return,
a [`MovementMode`](@ref) specification, and a `Solver` to solve 
linear equations in the problem.

This lazy specification allows ConScape to minimise the work
required to calculate multiple outputs: habitat connectivity
betweenness metrics etc can use the same memory allocations and solves.

# Keywords

- `measures`: A NamedTuple of [`GraphMeasure`](@ref)s.
- `movement_mode`: A [`MovementMode`](@ref).
- `solver`: A [`Solver`](@ref) specification, `VectorSolver` by default.
    [`LinearSolver`](@ref) allows for the use of any LinearSolve.jl solvers.
"""
@kwdef struct Problem{MM<:MovementMode,M,S<:Solver,C} <: AbstractProblem
    movement_mode::MM = RandomisedShortestPath()
    measures::M = ()
    solver::S = VectorSolver()
    costfunction::C = MinusLog()
    grain::Union{Nothing,Int} = nothing
end
Problem(measures::Union{Tuple,NamedTuple}; kw...) = 
    Problem(; measures, kw...)

# function Base.show(io, mime, p::Problem; indent="")
    # println(io, typeof(p).name.wrapper)
    # println(io, indent, "measures:             ", p.measures)
    # println(io, indent, "movement_mode:        ", p.movement_mode)
    # println(io, indent, "costs:                ", p.costs)
    # println(io, indent, "solver:               ", p.solver)
    # println(io, indent, "diagvalue:            ", typeof(p.diagvalue))
    # println(io, indent, "prune:                ", p.prune)
# end

movement_mode(p::Problem) = p.movement_mode
measures(p::Problem) = p.measures
solver(p::Problem) = p.solver
grain(p::Problem) = p.grain
costfunction(p::Problem) = p.costfunction
proximity_measure(p::Problem) = proximity_measure(movement_mode(p))
distance_transformation(p::Problem) = distance_transformation(movement_mode(p))
diagvalue(p::Problem) = diagvalue(movement_mode(p))
theta(p::Problem) = theta(movement_mode(p))

# Solve just calls `init` and `solve`
solve(p::Problem, input::Union{Grid,RasterStack}; kw...) = solve(init(p, input; kw...))
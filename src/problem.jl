# Recusive getters for nested problems
graph_measures(p::AbstractProblem) = graph_measures(p.problem)
connectivity_measure(p::AbstractProblem) = connectivity_measure(p.problem)
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
isthreaded(p::AbstractProblem) = false

"""
    Problem(graph_measures...; solver, θ)

A `Problem` specifies graph and connectivity measures,
and a method to solve them.

This lazy specification allows ConScape to minimise the work
required to calculate multiple outputs: habitat conectivity
betweenness metrics etc can use the same memory allocations and solves.

# Keywords

- `graph_measures`: A NamedTuple of [`GraphMeasure`](@ref)s.
- `connectivity_measure`: A [`ConnectivityMeasure`](@ref).
- `solver`: A [`Solver`](@ref) specification.
"""
@kwdef struct Problem{MM<:MovementMode,GM,SM<:Solver,CO} <: AbstractProblem
    movement_mode::MM
    graph_measures::GM
    solver::SM = VectorSolver()
    costfunction::CO = MinusLog()
end
Problem(graph_measures::Union{Tuple,NamedTuple}; kw...) = Problem(; graph_measures, kw...)

function Base.show(io, mime, p::Problem; indent="")
    println(io, typeof(p).name.wrapper)
    # println(io, indent, "graph_measures:       ", p.graph_measures)
    # println(io, indent, "connectivity_measure: ", p.connectivity_measure)
    # println(io, indent, "costs:                ", p.costs)
    # println(io, indent, "solver:               ", p.solver)
    # println(io, indent, "diagvalue:            ", typeof(p.diagvalue))
    # println(io, indent, "prune:                ", p.prune)
end

movement_mode(p::Problem) = p.movement_mode
connectivity_measure(p::Problem) = connectivity_measure(movement_mode(p))
distance_transformation(p::Problem) = distance_transformation(movement_mode(p))
diagvalue(p::Problem) = diagvalue(movement_mode(p))
graph_measures(p::Problem) = p.graph_measures
solver(p::Problem) = p.solver
costfunction(p::Problem) = p.costfunction

# Solve just calls `init` and `solve!`
solve(p::Problem, rast::RasterStack; kw...) = solve!(init(p, rast; kw...); kw...)
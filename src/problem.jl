# Recusive getters for nested problems
graph_measures(p::AbstractProblem) = graph_measures(p.problem)
connectivity_measure(p::AbstractProblem) = connectivity_measure(p.problem)
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
solver(p::AbstractProblem) = solver(p.problem)
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
@kwdef struct Problem{CM<:ConnectivityMeasure,MM<:MovementMode,GM,SM<:Solver,DV,CO} <: AbstractProblem
    connectivity_measure::CM
    movement_mode::MM
    graph_measures::GM
    solver::SM = VectorSolver()
    costs::CO = MinusLog()
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
connectivity_measure(p::Problem) = p.connectivity_measure
graph_measures(p::Problem) = p.graph_measures
solver(p::Problem) = p.solver
costs(p::Problem) = p.costs

# Solve just calls `init` and `solve!`
solve(p::Problem, rast::RasterStack; kw...) = solve!(init(p, rast; kw...), p; kw...)
# Solve defers to specific solver methods in solvers.jl
solve!(gp::GridPrecalculations, p::Problem; kw...) =
    solve!(solver(p), gp, p; kw...)

init(p::Problem, rast::RasterStack; kw...) = init(movementmode(p), p, rast; kw...)
init!(gp::GridPrecalculations, p::Problem, rast::RasterStack; kw...) =
    init!(solver(p), gp, p, rast; kw...)

# Recusive getters for nested problems
graph_measures(p::AbstractProblem) = graph_measures(p.problem)
connectivity_measure(p::AbstractProblem) = connectivity_measure(p.problem)
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
solver(p::AbstractProblem) = solver(p.problem)
isthreaded(p::AbstractProblem) = false

"""
    assess(p::AbstractProblem, rast::RasterStack)

Assess the computational requirements of problem
`p` for `RasterStack` `rastr`. 

This can be used to indicate memory and time reequiremtents on a cluster.
"""
function assess end

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
@kwdef struct Problem{GM,CM<:ConnectivityMeasure,SM<:Solver,DV,CO} <: AbstractProblem
    graph_measures::GM
    connectivity_measure::CM = LeastCostDistance()
    solver::SM = MatrixSolver()
    diagvalue::DV = nothing
    costs::CO = MinusLog()
    prune::Bool = true
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

diagvalue(p::Problem) = p.diagvalue
graph_measures(p::Problem) = p.graph_measures
connectivity_measure(p::Problem) = p.connectivity_measure
solver(p::Problem) = p.solver
costs(p::Problem) = p.costs
prune(p::Problem) = p.prune
isthreaded(p::Problem) = p.threaded

# Solve just calls `init` and `solve!`
solve(p::Problem, rast::RasterStack; kw...) = solve!(init(p, rast; kw...), p; kw...)
# Solve defers to specific solver methods in solvers.jl
solve!(workspace::NamedTuple, p::Problem; kw...) =
    solve!(workspace, solver(p), connectivity_measure(p), p; kw...)

# `init`` calls `init!` on an empty workspace
init(p::AbstractProblem, args...; kw...) = init!((;), p, args...; kw...)
# init! requirements are conditional on solver and connectivity measure
# See solvers.jl
function init!(workspace::NamedTuple, p::Problem, rast::RasterStack; verbose=false, kw...)
    verbose && println("Initialising for $(solver(p))")
    init!(workspace, solver(p), connectivity_measure(p), p, rast; kw...)
end

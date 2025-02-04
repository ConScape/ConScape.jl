# Recusive getters for nested problems
graph_measures(p::AbstractProblem) = graph_measures(p.problem)
connectivity_measure(p::AbstractProblem) = connectivity_measure(p.problem)
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
solver(p::AbstractProblem) = solver(p.problem)

"""
    assess(p::AbstractProblem, g)

Assess the memory and solve requirements of problem
`p` on grid `g`. This can be used to indicate memory
and time reequiremtents on a cluster
"""
function assess end

"""
    Problem(graph_measures...; solver, θ)

Combine multiple solve operations into a single object, 
to be run in the same job.

# Keywords

- `graph_measures`: A NamedTuple of [`GraphMeasure`](@ref)s.
- `connectivity_measure`: A [`ConnectivityMeasure`](@ref).
- `solver`: A [`Solver`](@ref) specification.
"""
@kwdef struct Problem{GM,CM<:ConnectivityMeasure,SM<:Solver,DV,CO} <: AbstractProblem
    graph_measures::GM
    connectivity_measure::CM = LeastCostDistance()
    solver::SM= MatrixSolver()
    diagvalue::DV=nothing
    costs::CO=MinusLog()
    prune::Bool=true
end
Problem(graph_measures::Union{Tuple,NamedTuple}; kw...) = Problem(; graph_measures, kw...)

diagvalue(p::Problem) = p.diagvalue
graph_measures(p::Problem) = p.graph_measures
connectivity_measure(p::Problem) = p.connectivity_measure
solver(p::Problem) = p.solver
costs(p::Problem) = p.costs
prune(p::Problem) = p.prune


solve(p::Problem, rast::RasterStack; kw...) = solve!(init(p, rast; kw...), p; kw...)
solve!(workspace::NamedTuple, p::Problem; kw...) = 
    solve!(workspace, solver(p), connectivity_measure(p), p; kw...)

# Init is conditional on solver and connectivity measure
function init!(workspace::NamedTuple, p::Problem, rast::RasterStack; kw...)
    println("Initialising for $(solver(p))")
    init!(workspace, solver(p), connectivity_measure(p), p, rast; kw...)
end

init(p::AbstractProblem, args...; kw...) = init!((;), p, args...; kw...)
# Defined earlier in ConScape.jl for load order
# abstract type AbstractProblem end
@doc """
    Problem

Abstract supertype for ConScape problem specifications.
""" Problem

# Recusive getters for nested problems
graph_measures(p::AbstractProblem) = graph_measures(p.problem)
connectivity_measure(p::AbstractProblem) = connectivity_measure(p.problem)
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
solver(p::AbstractProblem) = solver(p.problem)

"""
    solve(problem, grid::Union{Grid,GridRSP})

Solve problem `o` for a grid.
"""
function solve end

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
@kwdef struct Problem{GM,CM<:ConnectivityMeasure,SM<:Solver} <: AbstractProblem
    graph_measures::GM
    connectivity_measure::CM = LeastCostDistance()
    solver::SM = MatrixSolver()
end
Problem(graph_measures::Union{Tuple,NamedTuple}; kw...) = Problem(; graph_measures, kw...)

graph_measures(p::Problem) = p.graph_measures
connectivity_measure(p::Problem) = p.connectivity_measure
solver(p::Problem) = p.solver

solve(p::Problem, g::Grid; workspace=nothing) =
    solve(p.solver, connectivity_measure(p), p, g; workspace)
function solve(p::Problem, rast::RasterStack; workspace=nothing)
    grid = isnothing(workspace) ? Grid(p, rast) : workspace.grid
    return solve(p, grid; workspace)
end

function init(p::Problem, rast::RasterStack)
    grid = Grid(p, rast)
    return (; grid, init(p, grid)...)
end
# Init is conditional on solver and connectivity measure
init(p::AbstractProblem, g::Grid) = init(solver(p), connectivity_measure(p), p, g)
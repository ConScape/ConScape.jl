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
distance_transformation(p::AbstractProblem) =
    distance_transformation(connectivity_measure(p))
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

solve(p::Problem, rast::RasterStack) = solve(p, Grid(p, rast))
solve(p::Problem, g::Grid) = solve(p.solver, connectivity_measure(p), p, g)

# @kwdef struct ComputeAssesment{P,M,T}
#     problem::P
#     mem_stats::M
#     totalmem::T
# end

# """
#     allocate(co::ComputeAssesment)

# Allocate memory required to run `solve` for the assessed ops.

# The returned object can be passed as the `allocs` keyword to `solve`.
# """
# function allocate(co::ComputeAssesment)
#     zmax = co.zmax
#     # But actually do this with GenericMemory using Julia v1.11
#     Z = Matrix{Float64}(undef, co.zmax) 
#     S = sparse(1:zmax[1], 1:zmax[2], 1.0, zmax...)
#     L = lu(S)
#     # Just return a NamedTuple for now
#     return (; Z, S, L)
# end

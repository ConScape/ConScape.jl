# Recusive getters for nested problems
measures(p::AbstractProblem) = measures(problem(p))
movement(p::AbstractProblem) = movement(problem(p))
proximity_measure(p::AbstractProblem) = proximity_measure(problem(p))
connectivity_function(p::AbstractProblem) =
    connectivity_function(proximity_measure(p))
isthreaded(p::AbstractProblem) = false

"""
    Problem(measures; movement, solver, θ)

A `Problem` collects all parameters required to generate one or multiple 
[`Measure`](@ref) outputs from a single `RasterStack` or `Grid` input.

This lazy specification allows ConScape to minimize the work
required to calculate multiple measures: connectivity, betweenness 
and sensitivity (etc) measures can use the same memory allocations,
sparse matrix factorizations, and solves.

## Keywords

- `measures`: A NamedTuple of [`Measure`](@ref)s.
- `movement`: A [`MovementMode`](@ref), [`RandomisedShortestPath`](@ref)
    by default.
- `solver`: A [`Solver`](@ref) specification, `VectorSolver` by default.
    [`LinearSolver`](@ref) allows for the use of any LinearSolve.jl solvers,
    when the LinearSolve.jl package is loaded.
- `grain::Int`: used to apply coarse_graining to target qualities, 
    to reduce computational requirements.

## Initialising and solving

Problems are solved with `solve`:

```julia
problem = ConScape.Problem(measures; movement)
rast = RasterStack((source_qualities=qualpath, affinities=affinitypath))
result = solve(problem, rast)
```

For interactive work you can use `init`:

```julia
# Init for all subgraphs of the grid made from rast
multiinit = init(problem, rast)
# Init for the first subgraph
singleinit = init(multiinit, 1)
# Init for a single target
targetinit = init(singleinit, 1)

And `solve` will work at each level:

```julia
solve(multiinit)
solve(singleinit)
solve(targetinit) 
```
"""
@kwdef struct Problem{MM<:MovementMode,M,S<:Solver} <: AbstractProblem
    movement::MM = RandomisedShortestPath()
    measures::M = ()
    solver::S = VectorSolver()
    grain::Union{Nothing,Int} = nothing # Better name here - target_density?
end
Problem(measure::Measure; kw...) = Problem(; measures=(measure,), kw...)
Problem(measures::Union{Tuple,NamedTuple}; kw...) = Problem(; measures, kw...)

# function Base.show(io, mime, p::Problem; indent="")
    # println(io, typeof(p).name.wrapper)
    # println(io, indent, "measures:             ", p.measures)
    # println(io, indent, "movement:        ", p.movement)
    # println(io, indent, "costs:                ", p.costs)
    # println(io, indent, "solver:               ", p.solver)
    # println(io, indent, "diagvalue:            ", typeof(p.diagvalue))
    # println(io, indent, "prune:                ", p.prune)
# end

movement(p::Problem) = p.movement
measures(p::Problem) = p.measures
solver(p::Problem) = p.solver
grain(p::Problem) = p.grain
costfunction(p::Problem) = costfunction(movement(p))
proximity_measure(p::Problem) = proximity_measure(movement(p))
distance_transformation(p::Problem) = distance_transformation(movement(p))
diagvalue(p::Problem) = diagvalue(movement(p))
theta(p::Problem) = theta(movement(p))
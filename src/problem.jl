# Recusive getters for nested problems
measures(p::AbstractProblem) = measures(problem(p))
movement_mode(p::AbstractProblem) = movement_mode(problem(p))
proximity_measure(p::AbstractProblem) = proximity_measure(problem(p))
connectivity_function(p::AbstractProblem) =
    connectivity_function(proximity_measure(p))
isthreaded(p::AbstractProblem) = false

"""
    Problem(measures; movement_mode, solver, θ)

A `Problem` collects all parameters required to generate
one or multile outputs from a single `RasterStack` or `Grid`.
input.

It may be nested in [`WindowedProblem`](@ref) or [`BatchProblem`](@ref) 
to run at larger scales.

This lazy specification allows ConScape to minimize the work
required to calculate multiple outputs: connectivity, betweenness 
and sensitivity (etc) measures can use the same memory allocations,
sparse matrix factorizations, and solves.

## Keywords

- `measures`: A NamedTuple of [`Measure`](@ref)s.
- `movement_mode`: A [`MovementMode`](@ref), [`RandomisedShortestPath`](@ref)
    by default.
- `solver`: A [`Solver`](@ref) specification, `VectorSolver` by default.
    [`LinearSolver`](@ref) allows for the use of any LinearSolve.jl solvers,
    when LinearSolve.jl is loaded.
- `costfunction`: A function to transform affinities to costs, usually
    a [`Transformation`](@ref) but custom function also work. 
    The default is [`MinusLog`](@ref).
- `grain::Int`: used to apply coarse_graining to target qualities, 
    to reduce computational requirements.

## Initialising and solving

Problems are solved with `solve`:

```julia
problem = ConScape.Problem(measures; movement_mode)
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
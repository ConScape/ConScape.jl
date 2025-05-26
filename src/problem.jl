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
- `costfunction`: A `Function` or [`Transformation`](@ref) to
    convert likelyhoods to costs.
- `likelyhoodfunction`: A `Function` or [`Transformation`](@ref) to
    convert costs to likelyhoods.
-  `neighbors`: Whether to use 8 (queen) or 4 (rook) neighbors when generating, 
    graphs from a two-dimensional raster. `ConScape.N8` by default, can be `ConScape.N4`.
    With a three-dimensional matrix, `neighbors` keyword is not used.
- `transition_weight`: `TargetWeight()` by default, using only the value of the
    destination pixel (neighbor). Can be `AverageWeight()` to use the average of
    both neighboring nodes. If a three-dimensional matrix is provided, 
    `transition_weight` is not used as the array values are already transitions.

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
@kwdef struct Problem{MM<:MovementMode,M,S<:Solver,G,CF,LF,N,NW} <: AbstractProblem
    movement::MM = RandomisedShortestPath()
    measures::M = (;)
    solver::S = VectorSolver()
    grain::G = nothing # Better name here - target_density?
    costfunction::CF = MinusLog()
    likelyhoodfunction::LF = nothing
    neighbors::N = N8
    transition_weight::NW = TargetWeight()
    function Problem(
        movement::MM, m::M, solver::S, grain::G, costfunction::CF, likelyhoodfunction::LF, neighbors::N, transition_weight::NW
    ) where {MM,M,S,G,CF,LF,N,NW}
        m1 = if m isa Measure
            NamedTuple{(Symbol(m),)}((m,))
        elseif m isa Tuple
            NamedTuple{map(Symbol, m)}(m)
        else
            m
        end
        return new{MM,typeof(m1),S,G,CF,LF,N,NW}(movement, m1, solver, grain, costfunction, likelyhoodfunction, neighbors, transition_weight)
    end
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
costfunction(p::Problem) = p.costfunction
likelihoodfunction(p::Problem) = p.likelyhoodfunction
neighbors(p::Problem) = p.neighbors
transition_weight(p::Problem) = p.transition_weight
proximity_measure(p::Problem) = proximity_measure(movement(p))
distance_transformation(p::Problem) = distance_transformation(movement(p))
diagvalue(p::Problem) = diagvalue(movement(p))
theta(p::Problem) = theta(movement(p))

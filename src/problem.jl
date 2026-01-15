# Recusive getters for nested problems
measures(p::AbstractProblem) = measures(problem(p))
movement(p::AbstractProblem) = movement(problem(p))
proximity_measure(p::AbstractProblem) = proximity_measure(problem(p))
connectivity_function(p::AbstractProblem) =
    connectivity_function(proximity_measure(p))
isthreaded(p::AbstractProblem) = false

"""
    ConScapeProblem(measures; movement, solver, θ)

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
- `solver`: A [`Solver`](@ref) specification, `ColumnSolver` by default.
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
- `stepweight`: `TargetWeight()` by default, using only the value of the
    destination pixel (neighbor). Can be `AverageWeight()` to use the average of
    both neighboring nodes. If a three-dimensional matrix is provided,
    `stepweight` is not used as the array values are already transitions.
- `mmap_path`: Directory path for storing memory-mapped matrix workspaces, or 
    `nothing` (default) to use in-memory arrays. When set, large intermediate matrices
    (Z, Y) are stored as memory-mapped files, reducing Julia RAM usage at the
    cost of some disk I/O. This is only important for `SensitivityAnalysis`, 
    `EigMax` and `EdgeBetweenness` measures that need full matrices allocated.
    Only use with fast local drives (SSD/NVMe); network storage will degrade 
    performance significantly. 

## Initialising and solving

Problems are solved with `solve`:

```julia
problem = ConScapeProblem(measures; movement)
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
struct ConScapeProblem{
    MM<:MovementMode,M<:Union{Measure,NamedTuple},S<:Solver,G<:Union{Int,Nothing},CF,LF,N,SW<:StepWeight,MP<:Union{Nothing,String}
} <: AbstractProblem
    movement::MM
    measures::M
    solver::S
    grain::G
    costfunction::CF
    likelyhoodfunction::LF
    neighbors::N
    stepweight::SW
    mmap_path::MP
    function ConScapeProblem(
        movement::MM, m::M, solver::S, grain::G, costfunction::CF,
        likelyhoodfunction::LF, neighbors::N, stepweight::NW, mmap_path::MP
    ) where {MM,M,S,G,CF,LF,N,NW,MP}
        m1 = if m isa Tuple
            NamedTuple{map(Symbol, m)}(m)
        else
            m
        end
        return new{MM,typeof(m1),S,G,CF,LF,N,NW,MP}(
            movement, m1, solver, grain, costfunction,
            likelyhoodfunction, neighbors, stepweight, mmap_path
        )
    end
end
ConScapeProblem(measures::Union{Measure,Tuple,NamedTuple}; kw...) =
    ConScapeProblem(; measures, kw...)
ConScapeProblem(measures::Union{Measure,Tuple,NamedTuple}, movement::MovementMode; kw...) =
    ConScapeProblem(; kw..., measures, movement)
function ConScapeProblem(;
    movement=RandomisedShortestPath(),
    measures=(;),
    solver=ColumnSolver(),
    grain=nothing,
    costfunction=MinusLog(),
    likelyhoodfunction=nothing,
    neighbors=N8,
    stepweight=TargetWeight(),
    mmap_path=nothing,
)
    ConScapeProblem(
        movement, measures, solver, grain, costfunction, 
        likelyhoodfunction, neighbors, stepweight, mmap_path
    )
end

function Base.show(io::IO, mime::MIME"text/plain", p::ConScapeProblem; indent="")
    println(io, typeof(p).name.wrapper)
    println(io)
    println(io, indent, "measures:             ", measures(p))
    println(io, indent, "movement:             ", movement(p))
    println(io, indent, "costfunction:         ", costfunction(p))
    println(io, indent, "likelihoodfunction:   ", likelihoodfunction(p))
    println(io, indent, "solver:               ", solver(p))
    println(io, indent, "grain:                ", grain(p))
    println(io, indent, "stepweight:           ", stepweight(p))
    nbrs = if neighbors(p) == N8 
        "N8"
    elseif neighbors(p) == N4 
        "N4"
    else
        "custom"
    end
    println(io, indent, "neighbors:            ", nbrs)
    if !isnothing(mmap_path)
        println(io, indent, "mmap_path:        ", mmap_path(p))
    end
end

movement(p::ConScapeProblem) = p.movement
measures(p::ConScapeProblem) = p.measures
solver(p::ConScapeProblem) = p.solver
grain(p::ConScapeProblem) = p.grain
costfunction(p::ConScapeProblem) = p.costfunction
likelihoodfunction(p::ConScapeProblem) = p.likelyhoodfunction
neighbors(p::ConScapeProblem) = p.neighbors
stepweight(p::ConScapeProblem) = p.stepweight
mmap_path(p::ConScapeProblem) = p.mmap_path
proximity_measure(p::ConScapeProblem) = proximity_measure(movement(p))
distance_transformation(p::ConScapeProblem) = distance_transformation(movement(p))
diagvalue(p::ConScapeProblem) = diagvalue(movement(p))
theta(p::ConScapeProblem) = theta(movement(p))

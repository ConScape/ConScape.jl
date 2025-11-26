"""
    ReturnTrait

Traits that determing what object is allocated to store 
the return values of `compute` on a [`Measures`](@ref).
"""
abstract type ReturnTrait end
abstract type ReturnDenseSpatial <: ReturnTrait end
abstract type ReturnSparseGraph <: ReturnTrait end
struct ReturnDenseSpatialSum <: ReturnDenseSpatial end
struct ReturnAssignedSparse <: ReturnSparseGraph end
struct ReturnScalarSum <: ReturnTrait end
struct ReturnCustom <: ReturnTrait end

"""
    Level

Traits for specifying the level of a computation or output:
`GridGraphLevel` for the whole gridgraph, `ConnectedGraphLevel` for a 
connected subgraph, and `TargetLevel` for a single target pixel.
"""
abstract type Level end
struct TargetLevel <: Level end
struct ConnectedGraphLevel <: Level end
struct GridGraphLevel <: Level end

#=======================================================================================
Output allocation, update, and finalisation

[ `solve!` starts ]
    │
    ├─► [ 1. `allocate_output` ]
    │   │
    │   │   Role: Creates the empty containers (Rasters, Matrices, etc.) to hold results.
    │   │
    │   │   When: Called at the beginning of the process, primarily within the
    │   │         `GridGraphInit` to create the final output rasters, but also
    │   │         within `ConnectedGraphInit` for intermediate results.
    │   │
    │   └─► The empty containers are passed down through the `Init` objects.
    │
    ├─► [ `TargetInit` level computation ]
    │   │
    │   ├─► [ 2. `update_output!` ]
    │   │   │
    │   │   │   Role: Populates the containers with computed data.
    │   │   │
    │   │   │   When: Called repeatedly inside `_solve_single_target!`. After each
    │   │   │         measure is calculated for a single target, this function
    │   │   │         writes the resulting vector into the correct slice/column
    │   │   │         of the allocated output container.
    │   │   │
    │   │   └─► The container for the `ConnectedGraphInit` is now partially or fully filled.
    │   │
    │   └─► All targets for a `ConnectedGraphInit` are processed.
    │
    ├─► [ 3. `finalize_output!` ]
    │   │
    │   │   Role: Performs aggregation or final calculations on a completed container.
    │   │
    │   │   When: Called inside `_solve_all_targets!` after the loop over all targets
    │   │         is finished. It's used for measures that require all target-specific
    │   │         data to be present before a final value can be computed (e.g., eigenvector centrality).
    │   │
    │   └─► The `ConnectedGraphInit`'s output container is now finalized.
    │
    └─► [ 4. `transfer_output!` ]
        │
        │   Role: Copies finalized data from a smaller container to a larger one.
        │
        │   When: Called inside `_solve_all_subgraphs!` after a `ConnectedGraphInit`
        │         is fully solved. It takes the results from the "island" and copies
        │         them into the correct spatial location within the main `GridGraphInit`
        │         output rasters.
        ▼
[ `solve!` finishes, returns final output ]
=#

##########################################################################
# allocate_output
#
"""
    allocate_output!(::ConnectedGraphInit, [intermediates])::Pair

# Preallocate the output for a measure, depending on the output level and returntrait.
"""
allocate_output(l::Level, problem::ConScapeProblem, args...) = 
    allocate_output(l, measures(problem), problem, args...)
allocate_output(l::Level, measures::Union{Tuple,NamedTuple}, args...) = 
    map(m -> allocate_output(l, m, args...), measures)
allocate_output(l::Level, m::Measure, args...)::Pair = 
    allocate_output(l, returntrait(m), m, args...)

allocate_output(l::Level, rt::ReturnTrait, m::Measure, ggi::GridGraphInit)::Pair = 
    allocate_output(l, rt, m, problem(ggi), gridgraph(ggi), connectedgraphs(ggi))
function allocate_output(
    l::GridGraphLevel,
    ::ReturnScalarSum,
    ::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    GridGraphLevel() => zeros(Float64, length(connectedgraphs))
end
# We need to zero out all connected subgraphs
function allocate_output(
    l::GridGraphLevel,
    ::ReturnDenseSpatial,
    ::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    A = fill(NaN, size(gridgraph))
    return l => A
end
# We need to use output size specific to Level
function allocate_output(
    l::GridGraphLevel,
    ::ReturnSparseGraph,
    ::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    connectedgraphs::Vector
) 
    l => spzeros(Float64, gridgraph_size(gridgraph))
end

allocate_output(l::Level, m::Measure, cgi::ConnectedGraphInit)::Pair = 
    allocate_output(l, returntrait(m), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))
function allocate_output(
    l::Union{ConnectedGraphLevel,TargetLevel}, 
    ::ReturnDenseSpatial, 
    ::Measure,
    ::ConScapeProblem, 
    gridgraph::GridGraph, 
    connectedgraph::ConnectedGraph, 
    precalculation,
)
    A = fill(NaN, size(gridgraph))
    # Initialise pixels in the connected subgraph
    A[sourceids(connectedgraph)] .= 0.0
    return l => A
end
# 
function allocate_output(
    l::Union{ConnectedGraphLevel,TargetLevel},
    ::ReturnScalarSum,
    ::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    l => Ref(0.0)
end
# Use a zeroed out W matrix so the indices match
function allocate_output(
    l::ConnectedGraphLevel,
    ::ReturnSparseGraph,
    ::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
    precalculation
) 
    l => spzeros(Float64, connectedgraph_size(connectedgraph))
end
function allocate_output(
    l::TargetLevel,
    ::ReturnSparseGraph,
    ::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    A = fill(NaN, size(gridgraph))
    return l => A
end

allocate_intermediate(measure, cgi) = nothing

##########################################################################
# update_output!
#
# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure and the output level.
# Separating this from `compute` allows us to return different types of output 
# from the same computation, and to reuse the code accross multiple measures.

"""
    update_output!(::ConnectedGraphInit, [intermediates])

Copy or add the result of `compute` on the targets to the connected graph output.
"""
update_output!(output::Pair, gm::Measure, init::Initialisation, v) = 
    update_output!(output[2], output[1], gm, init, v) 
# By default update_output!based on the return trait
update_output!(output, level::Level, gm::Measure, init::Initialisation, v) = 
    update_output!(output, level, returntrait(gm), init, v) 
# Spatial outputs are always the same shape, independent of Level
update_output!(output::AbstractMatrix, ::Level, ::ReturnDenseSpatialSum, ti::TargetInit, v::AbstractVector) = 
    view(output, sourceids(ti)) .+= v
# SumScalar is always a single Ref, independent of Level 
update_output!(output::Ref, ::Level, ::ReturnScalarSum, ::Initialisation, v::Number) = output[] += v
# AssignSparse varies by Level
update_output!(output::AbstractMatrix, ::TargetLevel, ::ReturnAssignedSparse, ti::TargetInit, v::AbstractVector) = 
    output[sourceids(ti)] .= v
update_output!(output::AbstractMatrix, ::ConnectedGraphLevel, ::ReturnAssignedSparse, ti::TargetInit, v::AbstractVector) = 
    output[:, target(ti).connectedgraphidx] .= v

"""
    finalise_output!(::ConnectedGraphInit, [intermediates])

Apply any modifications required after all targets contribute
to the output.
"""
finalize_output!(cgi, intermediates) = 
    finalize_output!(outputs(cgi), measures(cgi), cgi, intermediates)
function finalize_output!(outputs::NamedTuple, measures::NamedTuple, cgi, intermediates)
    return map(outputs, measures, intermediates) do output, measure, intermediate
        finalize_output!(output, measure, cgi, intermediate)
    end
end
# Trivial default, just returns the output object as-is
finalize_output!((level, output)::Pair, m::Measure, sgi::ConnectedGraphInit, intermediates) = 
    finalize_output!(output, level, m, sgi, intermediates)
finalize_output!(output, level::Level, m::Measure, cgi::ConnectedGraphInit, intermediates) = nothing


"""
    transfer_output!(outputs::NamedTuple, cgi::ConnectedGraphInit)

Transfer output from `ConnectedGraphInit` to `GridGraphInit`.

This step is necessary because the GridGraph may no be a single connected
graph, or may simply have empty areas that we want to skip as an optimisation.

Either way the sparse matrices of `ConnectedGraphInit` are usually smaller than 
the sparse matrices in `GridGraphInit`, so we need to map between them.
"""
transfer_output!(outputs::NamedTuple, cgi::ConnectedGraphInit) = 
    transfer_output!(outputs, ConScape.outputs(cgi), measures(cgi), cgi) 
function transfer_output!(dest::NamedTuple, source::NamedTuple, measures::NamedTuple, cgi)
    map(dest, source, measures) do d, s, m
        transfer_output!(d, s, m, cgi)
    end
end
transfer_output!(d::Pair, s::Pair, m::Measure, cgi::ConnectedGraphInit) =
    transfer_output!(d[2], s[2], m, cgi)
transfer_output!(d, s, m::Measure, cgi::ConnectedGraphInit) =
    transfer_output!(d, s, returntrait(m), cgi)
# transfer_output!(dest, source, ::ReturnTrait, cgi::ConnectedGraphInit) = 
    # dest[connectedgraphid(cgi)] = source
transfer_output!(dest::Vector{T}, source::Ref{T}, ::ReturnScalarSum, cgi::ConnectedGraphInit) where T =
    dest[] = source[]
transfer_output!(dest::Vector, source, ::ReturnCustom, cgi::ConnectedGraphInit) =
    dest[connectedgraphid(cgi)] = source
transfer_output!(dest::AbstractMatrix, source::AbstractMatrix, ::ReturnDenseSpatialSum, cgi::ConnectedGraphInit) = 
    dest[sourceids(cgi)] .= source[sourceids(cgi)]
transfer_output!(dest, source, ::ReturnAssignedSparse, cgi::ConnectedGraphInit) = 
    dest[view(LinearIndices(size(cgi)), sourceids(cgi)), map(x -> x.gridgraphidx, targetids(cgi))] .+= source

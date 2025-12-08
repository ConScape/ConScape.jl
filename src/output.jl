struct MeasureOutput{M,O}
    measure::M
    output::O
end

"""
    allocate_gridgraph_output!(::ConnectedGraphInit)::Pair

# Preallocate the output for a measure, depending on the output level and returntrait.
"""
allocate_gridgraph_output(problem::ConScapeProblem, args...) =
    allocate_gridgraph_output(measures(problem), problem, args...)
allocate_gridgraph_output(measures::Union{Tuple,NamedTuple}, args...) =
    map(m -> allocate_gridgraph_output(m, args...)::MeasureOutput, measures)
allocate_gridgraph_output(m::Measure, args...) =
    allocate_gridgraph_output(returntrait(m), m, args...)

allocate_gridgraph_output(rt::ReturnTrait, m::Measure, ggi::GridGraphInit)::Pair =
    allocate_gridgraph_output(rt, m, problem(ggi), gridgraph(ggi), connectedgraphs(ggi))
function allocate_gridgraph_output(
    ::ReturnScalarSum,
    m::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    return MeasureOutput(m, zeros(Float64, length(connectedgraphs)))
end
# We need to zero out all connected subgraphs
function allocate_gridgraph_output(
    ::ReturnSpatial,
    m::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    o = fill(NaN, size(gridgraph))
    return MeasureOutput(m, o)
end
# We need to use output size specific to Level
function allocate_gridgraph_output(
    ::ReturnSparseGraph,
    m::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    return MeasureOutput(m, spzeros(Float64, gridgraph_size(gridgraph)))
end

allocate_connectedgraph_output(l::Level, problem::ConScapeProblem, args...) =
    allocate_connectedgraph_output(l, measures(problem), problem, args...)
allocate_connectedgraph_output(l::Level, measures::Union{Tuple,NamedTuple}, args...) =
    map(m -> allocate_connectedgraph_output(l, m, args...)::MeasureOutput, measures)
allocate_connectedgraph_output(l::Level, m::Measure, args...)::MeasureOutput =
    allocate_connectedgraph_output(l, returntrait(m), m, args...)
allocate_connectedgraph_output(l::Level, m::Measure, cgi::ConnectedGraphInit)::MeasureOutput =
    allocate_connectedgraph_output(l, returntrait(m), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))
function allocate_connectedgraph_output(
    l::Union{ConnectedGraphLevel,TargetLevel},
    ::ReturnSpatial,
    m::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    connectedgraph::ConnectedGraph,
    precalculation,
)
    o = fill(NaN, size(gridgraph))
    # Initialise pixels in the connected subgraph
    o[sourceids(connectedgraph)] .= 0.0
    return MeasureOutput(m, o)
end
function allocate_connectedgraph_output(
    l::Union{ConnectedGraphLevel,TargetLevel},
    ::ReturnScalarSum,
    m::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    o = Ref(0.0)
    return MeasureOutput(m, o)
end
# Use a zeroed out W matrix so the indices match
function allocate_connectedgraph_output(
    l::ConnectedGraphLevel,
    ::ReturnSparseGraph,
    m::Measure,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
    precalculation
)
    o = spzeros(Float64, connectedgraph_size(connectedgraph))
    return MeasureOutput(m, o)
end
function allocate_connectedgraph_output(
    l::TargetLevel,
    ::ReturnSparseGraph,
    m::Measure,
    ::ConScapeProblem,
    gridgraph::GridGraph,
    ::ConnectedGraph,
    precalculation
)
    o = fill(NaN, size(gridgraph))
    return MeasureOutput(m, o)
end

##########################################################################
# update_connectedgraph_output!
#
# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure and the output level.
# Separating this from `compute` allows us to return different types of output
# from the same computation, and to reuse the code accross multiple measures.

"""
    update_connectedgraph_output!(::ConnectedGraphInit)

Copy or add the result of `compute` on the targets to the connected graph output.
"""
update_connectedgraph_output!(output::Pair, gm::Measure, init::Initialisation, v) =
    update_connectedgraph_output!(output..., gm, init, v)
# By default update_connectedgraph_output!based on the return trait
update_connectedgraph_output!(output, level::Level, gm::Measure, init::Initialisation, v) =
    update_connectedgraph_output!(output, level, returntrait(gm), init, v)
# Spatial outputs are always the same shape, independent of Level
function update_connectedgraph_output!(output::AbstractMatrix, ::Level, ::ReturnSpatialTargetSum, ti::TargetInit, v::AbstractVector)
    view(output, sourceids(ti)) .+= v
    return output
end
function update_connectedgraph_output!(output::AbstractMatrix, ::Level, ::ReturnSpatialSourceSum, ti::TargetInit, v::AbstractVector)
    output[targetspatialidx(ti)] += sum(v)
    return output
end
function update_connectedgraph_output!(output::AbstractMatrix, l::Level, ::ReturnSpatialSourceAndTargetSum, ti::TargetInit, v::AbstractVector)
    update_connectedgraph_output!(output, l, ReturnSpatialSourceSum(), ti, v)
    update_connectedgraph_output!(output, l, ReturnSpatialTargetSum(), ti, v)
    return output
end
# SumScalar is always a single Ref, independent of Level
function update_connectedgraph_output!(output::Ref, ::Level, ::ReturnScalarSum, ::Initialisation, v::Number)
    output[] += v
    return output
end
# AssignSparse varies by Level
function update_connectedgraph_output!(output::AbstractMatrix, ::TargetLevel, ::ReturnAssignedSparse, ti::TargetInit, v::AbstractVector)
    output[sourceids(ti)] .= v
    return output
end
function update_connectedgraph_output!(output::AbstractMatrix, ::ConnectedGraphLevel, ::ReturnAssignedSparse, ti::TargetInit, v::AbstractVector)
    output[:, target(ti).connectedgraphidx] .= v
    # Return the output workspace
    put!(workspaces(ti), v)
    return output
end

"""
    finalize_connectedgraph_output!(::ConnectedGraphInit)

Apply any modifications required after all targets contribute
to the output.
"""
finalize_connectedgraph_output!(cgi) =
    finalize_connectedgraph_output!(outputs(cgi), measures(cgi), cgi)
function finalize_connectedgraph_output!(
    outputs::NamedTuple, measures::NamedTuple, cgi
)
    return map(outputs, measures) do output, measure
        finalize_connectedgraph_output!(output, measure, cgi)
    end
end
# Trivial default, just returns the output object as-is
function finalize_connectedgraph_output!(
    (output, level)::Pair{<:Any,<:Level},
    m::Measure,
    sgi::ConnectedGraphInit
)
    finalize_connectedgraph_output!(output, level, m, sgi)
end
function finalize_connectedgraph_output!(
    output, level::Level, m::Measure, cgi::ConnectedGraphInit
)
    finalize_connectedgraph_output!(output, m, cgi)
end
function finalize_connectedgraph_output!(
    output, m::Measure, cgi::ConnectedGraphInit
)
    nothing
end


"""
    transfer_to_gridgraph_output!(outputs::NamedTuple, cgi::ConnectedGraphInit)

Transfer output from `ConnectedGraphInit` to `GridGraphInit`.

This step is necessary because the GridGraph may no be a single connected
graph, or may simply have empty areas that we want to skip as an optimisation.

Either way the sparse matrices of `ConnectedGraphInit` are usually smaller than
the sparse matrices in `GridGraphInit`, so we need to map between them.
"""
transfer_to_gridgraph_output!(outputs::NamedTuple, cgi::ConnectedGraphInit) =
    transfer_to_gridgraph_output!(outputs, ConScape.outputs(cgi), measures(cgi), cgi)
function transfer_to_gridgraph_output!(
    dest::NamedTuple{K}, source::NamedTuple{K}, measures::NamedTuple{K}, cgi
) where K
    map(dest, source, measures) do d, s, m
        transfer_to_gridgraph_output!(d, s, m, cgi)
    end
end
function transfer_to_gridgraph_output!(
    dest, source, m::Measure, cgi::ConnectedGraphInit
)
    transfer_to_gridgraph_output!(dest, source, returntrait(m), cgi)
end

# ReturnScalarSum measures sum connectedgraph outputs to a `Ref` or zero dimensional array.
function transfer_to_gridgraph_output!(
    dest::Vector{T},
    source::Ref{T},
    ::ReturnScalarSum,
    cgi::ConnectedGraphInit
) where T
    dest[] = source[]
    return dest
end
# TODO: should this exist for ReturnCustom?
# function transfer_to_gridgraph_output!(
#     dest::AbstractVector,
#     source::AbstractVector,
#     ::ReturnCustom,
#     cgi::ConnectedGraphInit
# )
#     dest[connectedgraphid(cgi)] .= source
# end
# Spatial measures copy the sourceids from the connected graph to the same
# ids at the gri graph level - filling in unconnected parts of the raster
function transfer_to_gridgraph_output!(
    dest::AbstractMatrix,
    source::AbstractMatrix,
    ::ReturnSpatial,
    cgi::ConnectedGraphInit
)
    dest[sourceids(cgi)] .= source[sourceids(cgi)]
    return dest
end
# Assigned sparse measures copy sparse data into a larger sparse matrix.
# TODO should this just be dense?
function transfer_to_gridgraph_output!(
    dest::AbstractMatrix,
    source::AbstractMatrix,
    ::ReturnAssignedSparse,
    cgi::ConnectedGraphInit
)
    # TODO: remove this allocation ?
    I = map(x -> x.gridgraphidx, targetids(cgi))
    V = view(LinearIndices(size(cgi), sourceids(cgi)), I)
    view(dest, V) .+= source

    return dest
end

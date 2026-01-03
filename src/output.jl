struct MeasureOutput{M,O}
    measure::M
    output::O
end

const ReturnAssigned = Union{ReturnAssignedSparse,ReturnAssignedDense}

"""
    allocate_gridgraph_output!(::ConnectedGraphInit)::Pair

# Preallocate the output for a measure, depending on the output level and returntrait.
"""
allocate_gridgraph_output(problem::ConScapeProblem, args...) =
    allocate_gridgraph_output(measures(problem), args...)
allocate_gridgraph_output(measures::Union{Tuple,NamedTuple}, args...) =
    map(m -> allocate_gridgraph_output(m, args...)::MeasureOutput, measures)
allocate_gridgraph_output(m::Measure, args...) =
    allocate_gridgraph_output(returntrait(m), m, args...)

allocate_gridgraph_output(rt::ReturnTrait, m::Measure, ggi::GridGraphInit) =
    allocate_gridgraph_output(rt, m, gridgraph(ggi), connectedgraphs(ggi))
function allocate_gridgraph_output(
    ::ReturnScalarSum,
    m::Measure,
    ::GridGraph,
    connectedgraphs::Vector
)
    return MeasureOutput(m, zeros(Float64, length(connectedgraphs)))
end
# We need to zero out all connected subgraphs
function allocate_gridgraph_output(
    ::ReturnSpatial,
    m::Measure,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    o = fill(NaN, size(gridgraph))
    return MeasureOutput(m, o)
end
function allocate_gridgraph_output(
    ::ReturnAssignedDense,
    m::Measure,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    o = Vector{Matrix{Float64}}(undef, length(connectedgraphs))
    return MeasureOutput(m, o)
end
function allocate_gridgraph_output(
    ::ReturnSparseGraph,
    m::Measure,
    gridgraph::GridGraph,
    connectedgraphs::Vector
)
    o = Vector{SparseMatrixCSC{Float64,Int}}(undef, length(connectedgraphs))
    return MeasureOutput(m, o)
end

allocate_connectedgraph_output(l::Level, problem::ConScapeProblem, args...) =
    allocate_connectedgraph_output(l, measures(problem), args...)
allocate_connectedgraph_output(l::Level, measures::Union{Tuple,NamedTuple}, args...) =
    map(m -> allocate_connectedgraph_output(l, m, args...)::MeasureOutput, measures)
allocate_connectedgraph_output(l::Level, m::Measure, args...)::MeasureOutput =
    allocate_connectedgraph_output(l, returntrait(m), m, args...)
allocate_connectedgraph_output(l::Level, m::Measure, cgi::ConnectedGraphInit)::MeasureOutput =
    allocate_connectedgraph_output(l, returntrait(m), m, gridgraph(cgi), connectedgraph(cgi))
function allocate_connectedgraph_output(
    l::Union{ConnectedGraphLevel,TargetLevel},
    ::ReturnSpatial,
    m::Measure,
    gridgraph::GridGraph,
    connectedgraph::ConnectedGraph,
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
    ::GridGraph,
    ::ConnectedGraph,
)
    o = Ref(0.0)
    return MeasureOutput(m, o)
end
# Return a sparse matrix for ReturnAssignedSparse at ConnectedGraphLevel
function allocate_connectedgraph_output(
    l::ConnectedGraphLevel,
    ::ReturnAssignedSparse,
    m::Measure,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
)
    o = spzeros(Float64, connectedgraph_size(connectedgraph))
    return MeasureOutput(m, o)
end
# Return a sparse vector for ReturnAssignedSparse at TargetLevel
function allocate_connectedgraph_output(
    l::TargetLevel,
    ::ReturnAssignedSparse,
    m::Measure,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
)
    o = spzeros(Float64, nsources(connectedgraph))
    return MeasureOutput(m, o)
end
# Return a dense matrix for ReturnAssignedDense at ConnectedGraphLevel
function allocate_connectedgraph_output(
    l::ConnectedGraphLevel,
    ::ReturnAssignedDense,
    m::Measure,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
)
    o = fill(NaN, connectedgraph_size(connectedgraph))
    return MeasureOutput(m, o)
end
# Return a dense vector for ReturnAssignedDense at TargetLevel
function allocate_connectedgraph_output(
    l::TargetLevel,
    ::ReturnAssignedDense,
    m::Measure,
    ::GridGraph,
    connectedgraph::ConnectedGraph,
)
    o = fill(NaN, nsources(connectedgraph))
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
function update_connectedgraph_output!(
    output::AbstractMatrix, ::Level, ::ReturnSpatialTargetSum, ti::TargetInit, v::AbstractVector
)
    view(output, sourceids(ti)) .+= v
    return output
end
function update_connectedgraph_output!(
    output::AbstractMatrix, ::Level, ::ReturnSpatialSourceSum, ti::TargetInit, v::AbstractVector
)
    output[targetspatialidx(ti)] += sum(v)
    return output
end
function update_connectedgraph_output!(
    output::AbstractMatrix, l::Level, ::ReturnSpatialSourceAndTargetSum, ti::TargetInit, v::AbstractVector
)
    update_connectedgraph_output!(output, l, ReturnSpatialSourceSum(), ti, v)
    update_connectedgraph_output!(output, l, ReturnSpatialTargetSum(), ti, v)
    return output
end
# SumScalar is always a single Ref, independent of Level
function update_connectedgraph_output!(
    output::Ref, ::Level, ::ReturnScalarSum, ::Initialisation, v::Number
)
    output[] += v
    return output
end
# AssignSparse varies by Level
function update_connectedgraph_output!(
    output::AbstractVector, ::TargetLevel, ::ReturnAssigned, ti::TargetInit, v::AbstractVector
)
    output[sourceids(ti)] .= v
    return output
end
function update_connectedgraph_output!(
    output::AbstractMatrix, ::ConnectedGraphLevel, ::ReturnAssigned, ti::TargetInit, v::AbstractVector
)
    output[:, target(ti).connectedgraphidx] .= v
    return output
end

"""
    finalize_connectedgraph_output!(::ConnectedGraphInit)

Apply any modifications required after all targets contribute
to the output.
"""
finalize_connectedgraph_output!(finallevel::Level, cgi::ConnectedGraphInit) =
    finalize_connectedgraph_output!(outputs(cgi), measures(cgi), finallevel, cgi)
function finalize_connectedgraph_output!(
    outputs::NamedTuple, measures::NamedTuple, finallevel::Level, cgi::ConnectedGraphInit
)
    return map(outputs, measures) do output, measure
        finalize_connectedgraph_output!(output, measure, finallevel, cgi)
    end
end
function finalize_connectedgraph_output!(
    output, m::Measure, level::Level, cgi::ConnectedGraphInit
)
    return nothing
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
# Fallback: we just copy to a Vector for each connected graph
function transfer_to_gridgraph_output!(
    dest::Vector{T},
    source::T,
    m::ReturnTrait,
    cgi::ConnectedGraphInit
) where T
    dest[connectedgraphid(cgi)] = source
    return dest
end

"""
    ReturnTrait

Traits for preallocated return values of [`Measures`](@ref).
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

##########################################################################
# allocate_output
#
# Preallocate the output for a measure, depending on the output level and returntrait.
allocate_output(l::Level, problem::ConScapeProblem, args...) = allocate_output(l, measures(problem), problem, args...)
allocate_output(l::Level, measures::Union{Tuple,NamedTuple}, args...) = 
    map(m -> allocate_output(l, m, args...), measures)
# By default allocate output based on the return trait
allocate_output(l::Level, m::Measure, ggi::GridGraphInit) = allocate_output(l, m, problem(ggi), gridgraph(ggi), connectedgraphs(ggi))
allocate_output(l::Level, m::Measure, cgi::ConnectedGraphInit) = allocate_output(l, m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))
allocate_output(l::Level, m::Measure, args...) = allocate_output(l, returntrait(m), args...)

allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, ::ReturnScalarSum, ::ConScapeProblem, ::GridGraph, ::ConnectedGraph, precalculation) = l => Ref(0.0)
allocate_output(l::GridGraphLevel, rt::ReturnScalarSum, ::ConScapeProblem, ::GridGraph, connectedgraphs::Vector) = GridGraphLevel() => zeros(Float64, length(connectedgraphs))

function allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, ::ReturnDenseSpatial, ::ConScapeProblem, gridgraph::GridGraph, connectedgraph::ConnectedGraph, precalculation)
    A = fill(NaN, size(gridgraph))
    # Initialise pixels in the connected subgraph
    A[sourceids(connectedgraph)] .= 0.0
    return l => A
end
# We need to zero out all connected subgraphs
function allocate_output(l::GridGraphLevel, rt::ReturnDenseSpatial, ::ConScapeProblem, gridgraph::GridGraph, connectedgraphs::Vector)
    A = fill(NaN, size(gridgraph))
    return l => A
end
# We need to use output size specific to Level
allocate_output(l::GridGraphLevel, rt::ReturnSparseGraph, ::ConScapeProblem, gridgraph::GridGraph, connectedgraphs::Vector) = 
    l => spzeros(Float64, gridgraph_size(gridgraph))
# Use a zeroed out W matrix so the indices match
allocate_output(l::ConnectedGraphLevel, ::ReturnSparseGraph, ::ConScapeProblem, ::GridGraph, connectedgraph::ConnectedGraph, precalculation) = 
    l => spzeros(Float64, connectedgraph_size(connectedgraph))
function allocate_output(l::TargetLevel, ::ReturnSparseGraph, ::ConScapeProblem, gridgraph::GridGraph, ::ConnectedGraph, precalculation)
    A = fill(NaN, size(gridgraph))
    return l => A
end

##########################################################################
# update_output!
#
# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure and the output level.
# Separating this from `compute` allows us to return different types of output 
# from the same computation, and to reuse the code accross multiple measures.

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

# Transfer output from ConnectedGraphInit to GridGraphInit
transfer_output!(outputs, cgi::ConnectedGraphInit) = 
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


# Return a RasterStack if all outputs are Raster
_maybe_rasterstack(ggi) = _maybe_rasterstack(measures(ggi), outputs(ggi), ggi)
function _maybe_rasterstack(measures, outputs, ggi)
    out = _maybe_raster(measures, outputs, ggi)
    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end

# Return a Raster where possible
function _maybe_raster(
    measures::Union{MeasureTuple,MeasureNamedTuple}, 
    outputs::Union{Tuple,NamedTuple}, 
    g::Initialisation
)
    map(measures, outputs) do measure, output
        _maybe_raster(returntrait(measure), output, dims(g); name=Symbol(measure))
    end
end
_maybe_raster(rt, x::Pair, g::Initialisation; kw...) = _maybe_raster(rt, x[2], g; kw...)
_maybe_raster(rt, x::Pair, g::Union{Tuple,Nothing}; kw...) = _maybe_raster(rt, x[2], g; kw...)
_maybe_raster(rt, rast::Raster, g::Initialisation; kw...) = rast
_maybe_raster(rt, mat::AbstractMatrix, g::Initialisation; kw...) =
    _maybe_raster(rt, mat, dims(g); kw...)
_maybe_raster(rt::ReturnDenseSpatial, mat::Matrix{T}, dims::Tuple; kw...) where T =
    Raster(mat, dims; missingval=T(NaN), kw...)
_maybe_raster(rt::ReturnDenseSpatial, vec::Vector{T}, dims::Tuple; kw...) where T<:Number =
    Raster(vec, dims; missingval=T(NaN), kw...)
_maybe_raster(rt, x, y; kw...) = x

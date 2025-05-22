"""
    ReturnTrait

Traits for preallocated return values of [`Measures`](@ref).
"""
abstract type ReturnTrait end
abstract type DenseSpatial <: ReturnTrait end
abstract type SparseGraph <: ReturnTrait end
struct SumDenseSpatial <: DenseSpatial end
struct AssignSparse <: SparseGraph end
struct SumScalar <: ReturnTrait end
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
allocate_output(l::Level, problem::Problem, args...) = allocate_output(l, measures(problem), args...)
allocate_output(l::Level, measures::Union{Tuple,NamedTuple}, args...) = 
    map(m -> allocate_output(l, m, args...), measures)
# By default allocate output based on the return trait
allocate_output(l::Level, m::Measure, args...) = allocate_output(l, returntrait(m), args...)

allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, ::SumScalar, ::GridGraph, ::ConnectedGraph, precalculation) = l => Ref(0.0)
allocate_output(l::GridGraphLevel, rt::SumScalar, ::GridGraph, connectedgraphs::Vector) = GridGraphLevel() => zeros(Float64, length(connectedgraphs))

function allocate_output(l::Union{ConnectedGraphLevel,TargetLevel}, ::DenseSpatial, gridgraph::GridGraph, connectedgraph::ConnectedGraph, precalculation)
    A = fill(NaN, size(gridgraph))
    # Initialise pixels in the connected subgraph
    A[sourceids(connectedgraph)] .= 0.0
    return l => A
end
# We need to zero out all connected subgraphs
function allocate_output(l::GridGraphLevel, rt::DenseSpatial, gridgraph::GridGraph, connectedgraphs::Vector)
    A = fill(NaN, size(gridgraph))
    return l => A
end
# We need to use output size specific to Level
allocate_output(l::GridGraphLevel, rt::SparseGraph, gridgraph::GridGraph, connectedgraphs::Vector) = 
    l => spzeros(Float64, gridgraph_size(gridgraph))
# Use a zeroed out W matrix so the indices match
allocate_output(l::ConnectedGraphLevel, ::SparseGraph, ::GridGraph, connectedgraph::ConnectedGraph, precalculation) = 
    l => spzeros(Float64, connectedgraph_size(connectedgraph))
function allocate_output(l::TargetLevel, ::SparseGraph, gridgraph::GridGraph, ::ConnectedGraph, precalculation)
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
update_output!(output::AbstractMatrix, ::Level, ::SumDenseSpatial, ti::TargetInit, v::AbstractVector) = 
    view(output, sourceids(ti)) .+= v
# SumScalar is always a single Ref, independent of Level
update_output!(output::Ref, ::Level, ::SumScalar, ::Initialisation, v::Number) = output[] += v
# AssignSparse varies by Level
update_output!(output::AbstractMatrix, ::TargetLevel, ::AssignSparse, ti::TargetInit, v::AbstractVector) = 
    output[sourceids(ti)] .= v
update_output!(output::AbstractMatrix, ::ConnectedGraphLevel, ::AssignSparse, ti::TargetInit, v::AbstractVector) = 
    output[:, target(ti).connectedgraphidx] .= v

# Transfer output from ConnectedGraphInit to GridGraphInit
transfer_output!(outputs, measures::NamedTuple, sgi::ConnectedGraphInit) = 
    transfer_output!(outputs, ConScape.outputs(sgi), measures, sgi) 
function transfer_output!(dest, source, measures::NamedTuple, sgi)
    map(dest, source, measures) do d, s, m
        transfer_output!(d, s, m, sgi)
    end
end
transfer_output!(d, s, m::Measure, sgi::ConnectedGraphInit) =
    transfer_output!(d, s, returntrait(m), sgi)
transfer_output!(dest, source, ::ReturnTrait, sgi::ConnectedGraphInit) = 
    dest[connectedgraphid(sgi)] = source
transfer_output!(dest, source::Ref, ::SumScalar, sgi::ConnectedGraphInit) = 
    dest[connectedgraphid(sgi)] = source[]
transfer_output!(dest, source, ::SumScalar, sgi::ConnectedGraphInit) = 
    output[sourceids(sgi)] .= v
transfer_output!(dest, source, ::SumDenseSpatial, sgi::ConnectedGraphInit) = 
    output[sourceids(sgi)] .= v
transfer_output!(dest, source, ::AssignSparse, sgi::ConnectedGraphInit) = 
    output[view(LinearIndices(size(sgi)), sourceids(sgi)), map(x -> x.graphidx, targetids(ti))] .+= v
transfer_output!(dest::Ref, source::Ref, ::SumScalar, ::Initialisation) = dest[] = source[]


# Return a RasterStack if all outputs are Raster
function _maybe_raster_return(measures, outputs, mgi)
    out = _maybe_raster(measures, outputs, mgi)
    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end

# Return a Raster where possible
function _maybe_raster(
    measures::Union{MeasureTuple,MeasureNamedTuple}, 
    mats::Union{Tuple,NamedTuple}, 
    g::Initialisation
)
    map(measures, mats) do measure, mat 
        _maybe_raster(returntrait(measure), mat, dims(g); name=Symbol(measure))
    end
end
_maybe_raster(rt, x::Pair, g::Initialisation; kw...) = _maybe_raster(rt, x[2], g; kw...)
_maybe_raster(rt, x::Pair, g::Tuple; kw...) = _maybe_raster(rt, x[2], g; kw...)
_maybe_raster(rt, mat::Raster, g::Initialisation; kw...) = mat
_maybe_raster(rt, mat::AbstractMatrix, g::Initialisation; kw...) =
    _maybe_raster(rt, mat, dims(g); kw...)
_maybe_raster(rt::DenseSpatial, mat::Matrix{T}, dims::Tuple; kw...) where T =
    Raster(mat, dims; missingval=T(NaN), kw...)
_maybe_raster(rt, vec::Vector{T}, dims::Tuple; kw...) where T =
    Raster(vec, dims; missingval=T(NaN), kw...)
_maybe_raster(rt, x, y; kw...) = x
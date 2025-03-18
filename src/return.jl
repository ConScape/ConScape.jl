"""
    ReturnTrait

Traits for preallocated return values of GraphMeasures.
"""
abstract type ReturnTrait end
abstract type DenseSpatial <: ReturnTrait end
struct AssignDenseSpatial <: DenseSpatial end
struct SumDenseSpatial <: DenseSpatial end
struct AssignSparse <: ReturnTrait end
struct AssignDense <: ReturnTrait end
struct SumScalar <: ReturnTrait end

# These allow calculation of return allocations
# returntrait(::EigMax) = ReturnsEigMaxTuple() # (n, m) -> n + m

# Preallocate the output for a graph measure, where needed
allocate_output(gm::Measure, grid::Precalculations) = 
    allocate_output(returntrait(gm), grid)
function allocate_output(::DenseSpatial, grid::Precalculations)
    A = fill(NaN, size(grid))
    A[source_ids(grid)] .= 0.0
    return A
end
allocate_output(::SumScalar, g::Precalculations) = Ref(0.0)
allocate_output(::AssignDense, g::Precalculations) = zeros(Float64, sparse_size(g))
allocate_output(::AssignSparse, g::Precalculations) = spzeros(Float64, sparse_size(g))
# allocate_output(::ReturnTrait, g::Precalculations) = nothing

# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure
update_output!(output, gm::Measure, v, tp::TargetPrecalculations) = 
    update_output!(output, returntrait(gm), v, tp) 
update_output!(output, ::AssignDenseSpatial, v::Number, tp) = output[target(tp).spatial] = v
update_output!(output, ::SumScalar, v::Number, tp) = output[] += v
update_output!(output, ::SumDenseSpatial, v::AbstractVector, tp) = output[source_ids(tp)] .+= v
update_output!(output, ::SumDenseSpatial, v::AbstractMatrix, tp) = output .+= v
update_output!(output, ::AssignDense, v::AbstractVector, tp) = output[:, target(tp).id] .= v
# Not sure this one makes sense
update_output!(output, ::AssignSparse, v::AbstractVector, tp) = output[LinearIndices(size(tp))[source_ids(tp)], target(tp).id] .= v
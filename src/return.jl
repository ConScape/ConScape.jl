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
allocate_output(problem::Problem, grid::Initialisation) = 
    allocate_output(measures(problem), grid)
allocate_output(measures::Union{Tuple,NamedTuple}, grid::Initialisation) = 
    map(m -> allocate_output(m, grid), measures)
allocate_output(m::Measure, grid::Initialisation) = 
    allocate_output(returntrait(m), grid)
function allocate_output(::DenseSpatial, grid::Initialisation)
    A = fill(NaN, size(grid))
    A[source_ids(grid)] .= 0.0
    return A
end
function allocate_output(::DenseSpatial, mgi::MultiGridInit)
    A = fill(NaN, size(mgi))
    # Set zeros for all subgrid sources
    for grid in subgrids(mgi)
        A[source_ids(grid)] .= 0.0
    end
    return A
end
allocate_output(::SumScalar, g::Initialisation) = Ref(0.0)
allocate_output(::AssignDense, g::Initialisation) = zeros(Float64, sparse_size(g))
allocate_output(::AssignSparse, g::Initialisation) = spzeros(Float64, sparse_size(g))
# allocate_output(::ReturnTrait, g::Initialisation) = nothing

# Preallocate the output for a graph measure, where needed
allocate_target_output(problem::Problem, grid::Initialisation) = 
    allocate_target_output(measures(problem), grid)
allocate_target_output(measures::Union{Tuple,NamedTuple}, grid::Initialisation) = 
    map(m -> allocate_target_output(m, grid), measures)
allocate_target_output(m::Measure, grid::Initialisation) = 
    allocate_target_output(returntrait(m), grid)
allocate_target_output(rt::DenseSpatial, grid::Initialisation) = allocate_output(rt, grid)
allocate_target_output(::SumScalar, g::Initialisation) = Ref(0.0)
allocate_target_output(::AssignDense, g::Initialisation) = zeros(Float64, length(g))
allocate_target_output(::AssignSparse, g::Initialisation) = spzeros(Float64, length(g))

# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure
update_output!(output, gm::Measure, v, tp::TargetInit) = 
    update_output!(output, returntrait(gm), v, tp) 
update_output!(output, ::AssignDenseSpatial, v::Number, tp) = output[target(tp).spatial] = v
update_output!(output, ::SumDenseSpatial, v::AbstractVector, tp) = view(output, source_ids(tp)) .+= v
update_output!(output, ::SumDenseSpatial, v::AbstractMatrix, tp) = output .+= v
update_output!(output, ::SumScalar, v::Number, tp) = output[] += v
# Not sure this one makes sense
update_output!(output::AbstractVector, ::AssignSparse, v::AbstractVector, tp) = 
    output[LinearIndices(size(tp))[source_ids(tp)]] .= v
update_output!(output::AbstractMatrix, ::AssignSparse, v::AbstractVector, tp) = 
    output[LinearIndices(size(tp))[source_ids(tp)], target(tp).grid_id] .= v
function update_output!(output::AbstractVector, ::AssignDense, v::AbstractVector, tp)
    lininds = LinearIndices(size(tp))
    @show length(output) sparse_size(tp) length(lininds)
    for (i, s) in enumerate(source_ids(tp))
        sourcerow = lininds[s]
        output[sourcerow] = v[i]
    end
    return output
end
function update_output!(output::AbstractMatrix, ::AssignDense, v::AbstractVector, tp)
    targetcol = target(tp).grid_id
    lininds = LinearIndices(size(tp))
    for (i, s) in enumerate(source_ids(tp))
        sourcerow = lininds[s]
        output[sourcerow, targetcol] = v[i]
    end
    return output
end
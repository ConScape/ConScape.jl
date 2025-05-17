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
allocate_output(problem::Problem, x::Initialisation) = 
    allocate_output(measures(problem), x)
allocate_output(measures::Union{Tuple,NamedTuple}, x::Initialisation) = 
    map(m -> allocate_output(m, x), measures)
allocate_output(m::Measure, x::Initialisation) = 
    allocate_output(returntrait(m), x)
function allocate_output(::DenseSpatial, x::Initialisation)
    A = fill(NaN, size(x))
    A[sourceids(x)] .= 0.0
    return A
end
function allocate_output(::DenseSpatial, mgi::ProblemInit)
    A = fill(NaN, size(mgi))
    # Set zeros for all subgraph sources
    for graph in subgraphs(mgi)
        A[sourceids(graph)] .= 0.0
    end
    return A
end
allocate_output(::SumScalar, ::Initialisation) = Ref(0.0)
allocate_output(::AssignDense, x::Initialisation) = zeros(Float64, sparse_size(x))
# allocate_output(::AssignSparse, x::Initialisation) = spzeros(Float64, sparse_size(x))
# allocate_output(::ReturnTrait, x::Initialisation) = nothing

# Preallocate the output for a graph measure, where needed
allocate_target_output(problem::Problem, x::Initialisation) = 
    allocate_target_output(measures(problem), x)
allocate_target_output(measures::Union{Tuple,NamedTuple}, x::Initialisation) = 
    map(m -> allocate_target_output(m, x), measures)
allocate_target_output(m::Measure, x::Initialisation) = 
    allocate_target_output(returntrait(m), x)
allocate_target_output(rt::DenseSpatial, x::Initialisation) = allocate_output(rt, x)
allocate_target_output(::SumScalar, ::Initialisation) = Ref(0.0)
allocate_target_output(::AssignDense, x::Initialisation) = zeros(Float64, length(x))
# allocate_target_output(::AssignSparse, x::Initialisation) = spzeros(Float64, length(x))

# Here we update single target results to the output object
# How that works exactly depends on the returntrait of each graph measure
update_output!(output, gm::Measure, v, tp::TargetInit) = 
    update_output!(output, returntrait(gm), v, tp) 
update_output!(output, ::AssignDenseSpatial, v::Number, tp) = output[target(tp).spatial] = v
update_output!(output, ::SumDenseSpatial, v::AbstractVector, tp) = view(output, sourceids(tp)) .+= v
update_output!(output, ::SumDenseSpatial, v::AbstractMatrix, tp) = output .+= v
update_output!(output, ::SumScalar, v::Number, tp) = output[] += v
# Not sure this one makes sense
update_output!(output::AbstractVector, ::AssignSparse, v::AbstractVector, tp) = 
    output[LinearIndices(size(tp))[sourceids(tp)]] .= v
update_output!(output::AbstractMatrix, ::AssignSparse, v::AbstractVector, tp) = 
    output[LinearIndices(size(tp))[sourceids(tp)], target(tp).graphidx] .= v
function update_output!(output::AbstractVector, ::AssignDense, v::AbstractVector, tp)
    lininds = LinearIndices(size(tp))
    for (i, s) in enumerate(sourceids(tp))
        sourcerow = lininds[s]
        output[sourcerow] = v[i]
    end
    return output
end
function update_output!(output::AbstractMatrix, ::AssignDense, v::AbstractVector, tp)
    targetcol = target(tp).graphidx
    lininds = LinearIndices(size(tp))
    for (i, s) in enumerate(sourceids(tp))
        sourcerow = lininds[s]
        output[sourcerow, targetcol] = v[i]
    end
    return output
end
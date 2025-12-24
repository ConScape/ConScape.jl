struct AssessmentWarnings
    sourcequality_nan_found::Bool
    targetquality_nan_found::Bool
end

function Base.:(|)(aw1::AssessmentWarnings, aw2::AssessmentWarnings)
    AssessmentWarnings(
        aw1.sourcequality_nan_found | aw2.sourcequality_nan_found,
        aw1.targetquality_nan_found | aw2.targetquality_nan_found,    
    )
end
function Base.:(&)(aw1::AssessmentWarnings, aw2::AssessmentWarnings)
    AssessmentWarnings(
        aw1.sourcequality_nan_found & aw2.sourcequality_nan_found,
        aw1.targetquality_nan_found & aw2.targetquality_nan_found,    
    )
end
Base.any(aw::AssessmentWarnings) = aw.sourcequality_nan_found | aw.targetquality_nan_found
Base.all(aw::AssessmentWarnings) = aw.sourcequality_nan_found & aw.targetquality_nan_found


struct MemoryEstimate
    vec_workspaces::Int
    mat_workspaces::Int
    sparse_matrices::Int
    lu_factorization::Int
    graph::Int
    quality_vectors::Int
    outputs::Int
    gridgraph::Int
    total::Int
end
function MemoryEstimate(a1, a2, a3, a4, a5, a6, a7, a8)
    total = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    MemoryEstimate(a1, a2, a3, a4, a5, a6, a7, a8, total)
end
MemoryEstimate() = MemoryEstimate(0, 0, 0, 0, 0, 0, 0, 0, 0)

"""
    ProblemAssessment

Abstract supertype for problem assessments.

These calculate the computation size of an 
`AbstractWindowedProblem` for a specific `RasterStack`.

As large assessments are expensive to compute, it is 
recommended to write them to disk using e.g. JSON3.jl:

```julia
using JSON3, ConScape
... # problem and rast definition
assessment = assess(problem, rast)
JSON3.write("assessment.json", assessment)
```

And later read them again, here for when the assessment
was for a nested windowed problem:
```
using JSON3, ConScape
assessment = JSON3.read("assessment.json", NestedAssessment)
````
"""
abstract type ProblemAssessment end

Base.size(a::ProblemAssessment) = a.size

"""
    WindowAssessment <: ProblemAssessment

Assessment of an AbstractWindowedProblem that holds
a `ConScapeProblem`.

# Fields
- `size::Tuple{Int,Int}`: the size of the input and output RasterStack
- `shape::Tuple{Int,Int}`: the shape of the windowing
- `njobs::Int`: the number of problem runs required to finish the problem
- `sparse_sizes::Vector{Tuple{Int,Int}}`: the sizes of each window
- `mask::Vector{Bool}`: Vector{Bool} where `true` values are jobs that need to be run.
- `indices::Vector{Int}`: the indices of `mask` that are `true`.
- `memory_estimate::Float64`: estimated peak memory usage in MB.
"""
@kwdef struct WindowAssessment <: ProblemAssessment
    size::Tuple{Int,Int}
    shape::Tuple{Int,Int}
    njobs::Int
    mask::Vector{Bool}
    indices::Vector{Int}
    warnings::AssessmentWarnings
    sparse_sizes::Vector{Tuple{Int,Int}}
    memory_estimate::MemoryEstimate
end

"""
    NestedAssessment <: ProblemAssessment

Assessment of a nested `AbstractWindowedProblem`,
that holds another `AbstractWindowedProblem`.

# Fields
- `size::Tuple{Int,Int}`: the size of the `RasterStack` input and output.
- `shape::Tuple{Int,Int}`: the shape of the windowing
- `njobs::Int`: the number of problem runs required to finish the problem
- `mask::Vector{Bool}`: Vector{Bool} where `true` values are jobs that need to be run.
- `indices::Vector{Int}`: the indices of `mask` that are `true`.
- `assessments::Vector{WindowAssessment}`: asessments at the next level down.
"""
@kwdef struct NestedAssessment <: ProblemAssessment
    size::Tuple{Int,Int}
    shape::Tuple{Int,Int}
    njobs::Int
    mask::Vector{Bool}
    indices::Vector{Int}
    warnings::AssessmentWarnings
    assessments::Vector{WindowAssessment}
end

function Base.show(io::IO, mime::MIME"text/plain", a::ProblemAssessment)
    println(io, typeof(a))
    println(io, "Shape: $(a.shape)")
    println(io, "Number of jobs: $(a.njobs)")
    println(io, "Memory estimate: $(round(a.memory_estimate.total / 2^20)) MB")  # Convert to MB
    # Use SparseArrays nice matrix printing for the mask
    if any(a.warnings)
        println(io, "Warnings: $(a.warnings)")
    end
    println(io, "Job mask: ")
    mask = sparse(reshape(a.mask, a.shape))
    Base.print_array(io, mask)
end


"""
    assess(p::AbstractProblem, rast::RasterStack)

Assess the computational requirements of problem
`p` for `RasterStack` `rastr`. 

This can be used to indicate memory and time reequiremtents on a cluster.
"""
function assess end

function assess(p::AbstractWindowedProblem{<:ConScapeProblem}, rast::AbstractRasterStack; 
    inner_target_bools=nothing,
    target_ranges=_target_ranges(p, rast),
    kw...
)
    # Define the ranges of each window
    window_ranges = ConScape.window_ranges(p, rast)

    # Convert everything to Bool at the batch level so window assessments are fast
    inner_targets = view(_get_targetquality(rast), target_ranges...)
    warnings = AssessmentWarnings(
        any(isnan, _get_sourcequality(rast)),
        any(isnan, inner_targets),
    )
    inner_target_bools = isnothing(inner_target_bools) ? _isvalid.(inner_targets) : inner_target_bools
    sourcequality = _isvalid.(_get_sourcequality(rast))
    targetquality = falses(size(rast))
    targetquality[target_ranges...] .= inner_target_bools
    bool_rast = RasterStack((; sourcequality, targetquality), dims(rast))

    # Calculate window sizes and allocations
    sparse_sizes = vec(_estimate_sparse_sizes(p, bool_rast; window_ranges))

    # Organise stats for each window into vectors
    window_mask = map(s -> prod(s) > 0, sparse_sizes)
    non_empty_indices = eachindex(window_mask)[window_mask]

    # Calculate global stats
    njobs = count(window_mask)
    shape = size(window_ranges)

    # Calculate memory estimate based on max window size
    memory_estimate = if njobs > 0
        _, max_idx = findmax(prod, sparse_sizes)
        max_sources, max_targets = sparse_sizes[max_idx]
        estimate_memory(problem(p), max_sources, max_targets)
    else
        MemoryEstimate(0, 0, 0, 0, 0, 0, 0, 0, 0)
    end

    WindowAssessment(size(rast), shape, njobs, window_mask, non_empty_indices, warnings, sparse_sizes, memory_estimate)
end
function assess(
    p::AbstractWindowedProblem{<:AbstractWindowedProblem},
    rast::AbstractRasterStack;
    nthreads=Threads.nthreads(),
    verbose=true,
    kw...
)
    # Calculate outer window ranges
    window_ranges = ConScape.window_ranges(p, rast)
    verbose && println("Assessing $(length(window_ranges)) jobs")

    # Define a vector for all assessment data
    assessments = Vector{WindowAssessment}(undef, length(window_ranges))

    # Run assessments threaded as they can take a long time for large rasters
    Threads.@threads for i in eachindex(vec(window_ranges))
        rs = window_ranges[i]
        verbose && println("Assessing batch: $i, $rs")
        function empty_assesment(size)
            verbose && println("  No targets found")
            WindowAssessment(;
                size,
                shape=(0, 0),
                njobs=0,
                mask=Bool[],
                indices=Int[],
                warnings=AssessmentWarnings(false, false),
                sparse_sizes=Tuple{Int,Int}[],
                memory_estimate=MemoryEstimate(),
            )
        end
        # We only need qualities for the assessment
        window_rast = RasterStack((
            sourcequality=_get_sourcequality(rast), 
            targetquality=_get_targetquality(rast),
        ))[rs...]
        target_ranges = _target_ranges(p, window_rast)
        # Convert targets to bool as early as possible
        inner_targets = view(_get_targetquality(window_rast), target_ranges...)
        inner_target_bools = _isvalid.(inner_targets)
        assessments[i] = if count(inner_target_bools) > 0
            assess(p.problem, window_rast; inner_target_bools, target_ranges, nthreads, kw...)
        else
            empty_assesment(size(window_rast))
        end
    end

    # Get mask and indices
    mask = map(a -> any(a.mask), assessments)
    non_empty_indices = eachindex(vec(mask))[mask]

    # Calculate global stats
    njobs = count(mask)
    shape = size(window_ranges)
    warnings = reduce(|, (a.warnings for a in assessments))

    return NestedAssessment(size(rast), shape, njobs, mask, non_empty_indices, warnings, assessments)
end

"""
    reassess(p::BatchProblem, a::NestedAssessment)

Re-asses an existing nested assessment of a [`BatchProblem`](@ref).

The returned `NestedAssessment` will have removed batches that 
already have a data folder (assumed to be successfully completed).
"""
function reassess(p::BatchProblem, a::NestedAssessment)
    patch = _reassess(p, a)
    a1 = ConstructionBase.setproperties(a, patch)
    # Update nan_target_found from remaining indices
    warnings = reduce(|, (a1.assessments[i].warnings for i in a1.indices); 
        init=AssessmentWarnings(false, false)
    )
    return ConstructionBase.setproperties(a1, (; warnings))
end
function reassess(p::BatchProblem, a::WindowAssessment)
    patch = _reassess(p, a)
    return ConstructionBase.setproperties(a, patch)
end

function _reassess(p, a)
    # Paths for all batches
    paths = batch_paths(p, size(a))
    # Paths for non-empty batches 
    jobpaths = paths[a.indices]
    # Find all the jobs that havent been saved (failed)
    idxmask = .!(isdir.(jobpaths))
    # Generate new arrays of indices and assessments for the remaining jobs
    indices = a.indices[idxmask]
    mask = fill(false, prod(a.shape))
    mask[indices] .= true
    njobs = length(indices)
    return (; njobs, mask, indices)
end

# Accept ProblemAssessment as an argument to solve and init 
# To used instead of keywords
solve(p::BatchProblem, rast::RasterStack, a::ProblemAssessment, i::Int; kw...) =
    solve(p, rast, i; _assessment_keywords(p, rast, a)..., kw...)
   
init(p::BatchProblem{<:WindowedProblem}, rast::RasterStack, a::NestedAssessment, i::Int...; kw...) =
    init(p, rast, i...; _assessment_keywords(p, rast, a)..., kw...)
init(p::BatchProblem{<:ConScapeProblem}, rast::RasterStack, a::WindowAssessment, i::Int...; kw...) =
    init(p, rast, i...; batch_indices=a.indices)
init(p::WindowedProblem{<:ConScapeProblem}, rast::RasterStack, a::WindowAssessment; kw...) =
    init(p, rast; sparse_sizes=a.sparse_sizes, indices=a.indices, kw...)

# Keywords to pass from an Assessment to `init` or `solve`
# We don't use the `Assesment` directly to allow manual manipulation
# of the batch via keywords.
function _assessment_keywords(::BatchProblem, rast, a::WindowAssessment)
    return (; batch_indices=a.indices)
end
function _assessment_keywords(p::BatchProblem, rast, a::NestedAssessment)
    sparse_sizes = map(a.assessments) do a_w
        a_w.sparse_sizes
    end
    window_indices = map(a.assessments) do a_w
        a_w.indices
    end
    return (; batch_indices=a.indices, window_indices, sparse_sizes)
end

batch_paths(p::BatchProblem, a::NestedAssessment) = batch_paths(p, size(a))[a.indices]


# Estimation tools #########################################################################

"""
    estimate_memory(problem::ConScapeProblem, sources::Int, targets::Int)

Estimate memory usage in bytes for a problem with given source and target counts.

Returns a NamedTuple with breakdown by component and total.

# Example

```julia
est = estimate_memory(problem, 1600, 400)
est.total          # Total bytes
est.mat_workspaces # Dense matrix vec_workspace bytes
```
"""
function estimate_memory(problem::ConScapeProblem, n_sources::Int, n_targets::Int)
    mes = measures(problem)
    mov = movement(problem)
    n_neighbors = length(neighbors(problem))

    sparse_matrix_bytes = _sizeofsparse(n_sources, n_sources, n_neighbors)
    dense_matrix_bytes = n_sources * n_targets * sizeof(Float64)

    vec_workspaces = num_vec_workspaces(problem) * n_sources * sizeof(Float64)
    mat_workspaces = num_mat_workspaces(problem) * n_sources * n_targets * sizeof(Float64)

    sparse_matrices = 8 * sparse_matrix_bytes
    lu_factorization = 2 * 4 * sparse_matrix_bytes
    graph = n_sources * n_neighbors * 2 * sizeof(Float64)
    quality_vectors = (2 * n_sources + n_targets) * sizeof(Float64)
    outputs = length(mes) * n_sources * sizeof(Float64)
    gridgraph = 2sparse_matrix_bytes

    return MemoryEstimate( 
        vec_workspaces,
        mat_workspaces,
        sparse_matrices,
        lu_factorization,
        graph,
        quality_vectors,
        outputs,
        gridgraph,
    )
end

function _sizeofsparse(n_rows::Int, n_cols::Int, n_neighbors::Int)
    n_cols * n_neighbors * sizeof(Float64) + # nzval
    n_cols * n_neighbors * sizeof(Int64) +   # rowval 
    (n_rows + 1) * sizeof(Int64) +           # colptr 
    sizeof(Int64) +                          # m
    sizeof(Int64)                            # n
end

"""
    estimate_memory_for_centersize(problem::ConScapeProblem, assessment::WindowAssessment, centersize::Int)

Quickly estimate memory requirements for a different `centersize` without re-running
the full assessment. This is approximate - it assumes the worst case where all
`centersize²` target cells in the center are valid.

This is useful for tuning `centersize` to fit memory constraints after running
an expensive `assess()` call on a large raster.

# Arguments
- `problem`: The ConScapeProblem (needed for vec_workspace counts)
- `assessment`: An existing WindowAssessment from `assess()`
- `centersize`: The new centersize to estimate memory for

# Returns
Memory estimate in MB for the new centersize.

# Example
```julia
# Run assessment once (expensive for large rasters)
assessment = assess(wp, rast)

# Quickly test different centersize values
for cs in [10, 15, 20, 25, 30]
    mem = estimate_memory_for_centersize(problem, assessment, cs)
    println("centersize=\$cs: \$(round(mem, digits=1)) MB")
end
```

# Memory scaling
For measures with dense matrices (EigMax, SensitivityAnalysis), memory scales as
`O(sources × targets)` where `targets = centersize²`. Halving centersize reduces
dense matrix memory by ~4x.

See also: [`assess`](@ref), [`estimate_memory`](@ref)
"""
function estimate_memory_for_centersize(
    problem::ConScapeProblem,
    assessment::WindowAssessment,
    centersize::Int
)
    # Get max sources from existing assessment (window size doesn't change much)
    max_sources = maximum(first, assessment.sparse_sizes; init=0)

    # New targets = centersize² (worst case: all center cells valid)
    new_targets = centersize^2

    # Recalculate memory estimate
    estimate_memory(problem, max_sources, new_targets).total / 1024^2
end


function _max_estimated_sparse_sizes(p::AbstractWindowedProblem, rast; kw...)
    sizes = _estimate_sparse_sizes(p, rast; kw...)
    _, i = findmax(prod, sizes)
    return sizes[i]
end

# Calculate the maximum number of source and target values in any window
function _estimate_sparse_sizes(p::AbstractWindowedProblem, rast;
    window_ranges=window_ranges(p, rast)
)
    # Calculate the maximum number of source and target values in any window
    return map(r -> _estimate_sparse_sizes(p, rast, r), window_ranges)
end

# This function extimates problem size without actually constructing grids.
# It cant be too small, but may be too large
_estimate_sparse_sizes(p::AbstractProblem, rast) = _estimate_sparse_sizes(p, rast, axes(rast))
function _estimate_sparse_sizes(p::AbstractProblem, rast, ranges::Tuple)
    source_count = _valid_sources(count, p, rast, ranges)
    target_count = _valid_targets(count, p, rast, ranges)
    return source_count, target_count
end

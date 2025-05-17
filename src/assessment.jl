
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
a `Problem`.

# Fields
- `size::Tuple{Int,Int}`: the size of the input and output RasterStack
- `shape::Tuple{Int,Int}`: the shape of the windowing
- `njobs::Int`: the number of problem runs required to finish the problem
- `sparse_sizes::Vector{Tuple{Int,Int}}`: the sizes of each window
- `mask::Vector{Bool}`: Vector{Bool} where `true` values are jobs that need to be run.
- `indices::Vector{Int}`: the indices of `mask` that are `true`.
"""
@kwdef struct WindowAssessment <: ProblemAssessment
    size::Tuple{Int,Int}
    shape::Tuple{Int,Int}
    njobs::Int
    mask::Vector{Bool}
    indices::Vector{Int}
    warnings::AssessmentWarnings
    sparse_sizes::Vector{Tuple{Int,Int}}
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

function assess(p::AbstractWindowedProblem{<:Problem}, rast::AbstractRasterStack; 
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

    WindowAssessment(size(rast), shape, njobs, window_mask, non_empty_indices, warnings, sparse_sizes)
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
init(p::BatchProblem{<:Problem}, rast::RasterStack, a::WindowAssessment, i::Int...; kw...) =
    init(p, rast, i...; batch_indices=a.indices)
init(p::WindowedProblem{<:Problem}, rast::RasterStack, a::WindowAssessment; kw...) =
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
"""
    ProblemAssessment

Abstract supertype for problem assessments.

These calculate the computation size of an 
`AbstractWindowedProblem` for a specific `RasterStack`.
"""
abstract type ProblemAssessment end

Base.size(a::ProblemAssessment) = a.size

"""
    WindowAssessment <: ProblemAssessment

Assessment of an AbstractWindowedProblem that holds
a `Problem`.

# Fields
- `shape::Tuple{Int,Int}`: the shape of the windowing
- `njobs::Int`: the number of problem runs required to finish the problem
- `sizes::Vector{Tuple{Int,Int}}`: the sizes of each window
- `mask::Vector{Bool}`: Vector{Bool} where `true` values are jobs that need to be run.
- `indices::Vector{Int}`: the indices of `mask` that are `true`.
"""
@kwdef struct WindowAssessment <: ProblemAssessment
    size::Tuple{Int,Int}
    shape::Tuple{Int,Int}
    njobs::Int
    mask::Vector{Bool}
    indices::Vector{Int}
    sizes::Vector{Tuple{Int,Int}}
end

"""
    NestedAssessment <: ProblemAssessment

Assessment of a nested `AbstractWindowedProblem`,
that holds another `AbstractWindowedProblem`.

# Fields
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
    assessments::Vector{WindowAssessment}
end

function Base.show(io::IO, mime::MIME"text/plain", bs::ProblemAssessment)
    println(io, "NestedAssessment")
    println(io)
    println(io, "Raster shape: $(bs.shape)")
    println(io, "Number of jobs: $(bs.njobs)")
    # Use SparseArrays nice matrix printing for the mask
    println(io, "Job mask: ")
    mask = sparse(reshape(bs.mask, bs.shape))
    Base.print_array(io, mask)
end


"""
    assess(p::AbstractProblem, rast::RasterStack)

Assess the computational requirements of problem
`p` for `RasterStack` `rastr`. 

This can be used to indicate memory and time reequiremtents on a cluster.
"""
function assess end

function assess(p::AbstractWindowedProblem{<:Problem}, rast::AbstractRasterStack; kw...)
    # Define the ranges of each window
    window_ranges = _window_ranges(p, rast)

    # Calculate window sizes and allocations
    window_sizes = map(vec(window_ranges)) do rs
        window_rast = view(rast, rs...)
        _problem_size(p, window_rast)
    end

    # Organise stats for each window into vectors
    window_mask = map(s -> prod(s) > 0, window_sizes)
    window_indices = eachindex(window_mask)[window_mask]

    # Calculate global stats
    njobs = count(window_mask)
    shape = size(window_ranges)

    WindowAssessment(size(rast), shape, njobs, window_mask, window_indices, window_sizes)
end
function assess(
    p::AbstractWindowedProblem{<:AbstractWindowedProblem},
    rast::AbstractRasterStack;
    nthreads=Threads.nthreads(),
    verbose=true,
    kw...
)
    # Calculate outer window ranges
    window_ranges = _window_ranges(p, rast)
    verbose && println("Assessing $(length(window_ranges)) jobs")

    # Define a channel to store window raster and reuse memory
    channel = Channel{Any}(Threads.nthreads())
    open(rast) do o
        for i in 1:nthreads
            put!(channel, _get_window_with_zeroed_buffer(getindex, p, o, first(window_ranges)))
        end
    end

    # Define a vector for all assessment data
    assessments = Vector{WindowAssessment}(undef, length(window_ranges))
    # Run assessments threaded as they can take a long time for large rasters
    Threads.@threads for i in eachindex(vec(window_ranges))
        rs = window_ranges[i]
        verbose && println("Assessing batch: $i, $rs")
        window_rast = take!(channel)
        function empty_assesment()
            verbose && println("  No targets found")
            WindowAssessment(;
                shape=(0, 0),
                njobs=0,
                sizes=Tuple{Int,Int}[],
                mask=Bool[],
                indices=Int[],
            )
        end
        # Just load the target window quickly first to avoid loading large rasters
        window_view = view(rast, rs...)
        quick_targets = window_view.target_qualities[_target_ranges(p, window_view)...]
        assessments[i] = if count(_isvalid, quick_targets) > 0
            # TODO 
            window_rast = open(rast) do o
                if map(length, rs) == size(window_rast)
                    _get_window_with_zeroed_buffer!(window_rast, p, o, rs)
                else
                    _get_window_with_zeroed_buffer(getindex, p, o, rs)
                end
            end
            nvalid = count(_isvalid, window_rast.target_qualities)
            if nvalid > 0
                verbose && println("  nvalid: $nvalid")
                assess(p.problem, window_rast; nthreads, kw...)
            else
                empty_assesment()
            end
        else
            empty_assesment()
        end
        put!(channel, window_rast)
    end
    # Get mask and indices
    mask = map(a -> any(a.mask), assessments)
    indices = eachindex(vec(mask))[mask]
    # Calculate global stats
    njobs = count(mask)
    shape = size(window_ranges)
    return NestedAssessment(size(rast), shape, njobs, mask, indices, assessments)
end

"""
    reassess(a::NestedAssessment, p::BatchProblem)

Re-asses an existing nested assesment of a BatchProblem.

The returned `NestedAssessment` will exclude any jobs that 
already have a data folder (assumed to be successfully completed).
"""
function reassess(p::BatchProblem, a::NestedAssessment)
    (; njobs, mask, indices) = _reassess(p, a)
    assessments = a.assessments[indices]
    return NestedAssessment(a.size, a.shape, njobs, mask, indices, assessments)
end
function reassess(p::BatchProblem, a::WindowAssessment)
    (; njobs, mask, indices) = _reassess(p, a)
    sizes = a.sizes[indices]
    return WindowAssessment(a.size, a.shape, njobs, mask, indices, sizes)
end

function _reassess(p, a)
    # Paths for all batches
    paths = _batch_paths(p, size(a))
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


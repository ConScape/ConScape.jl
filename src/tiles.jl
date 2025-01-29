# This file is a work in progress...

abstract type AbstractWindowedProblem{P} <: AbstractProblem end

costs(p::AbstractWindowedProblem) = costs(p.problem)
prune(p::AbstractWindowedProblem) = prune(p.problem)
buffer(p::AbstractWindowedProblem) = p.buffer
buffer(p::AbstractProblem) = 0
grain(::AbstractProblem) = nothing

"""
    WindowedProblem(problem::AbstractProblem; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.

`problem` is usually a [`Problem`](@ref) object but can be any `AbstractProblem`.

# Keywords

- `problem`: The radius of the window.
- `centersize`: the size of the target square.
- `buffer`: the area outside the source window.
- `threaded`: Whether to run in parallel. `false` by default
"""
@kwdef struct WindowedProblem{P} <: AbstractWindowedProblem{P}
    problem::P
    centersize::Int
    buffer::Int
    threaded::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

centersize(p::WindowedProblem) = p.centersize, p.centersize

function solve(p::WindowedProblem, rast::RasterStack; kw...)
    workspace = init(p, rast)
    solve!(workspace, p, rast; kw...)
end
function solve!(workspace, p::WindowedProblem, rast::RasterStack; 
    test_windows::Bool=false,
    verbose::Bool=false,
    mosaic_return::Bool=true,
    window_ranges=_window_ranges(p, rast),
    window_sizes=_window_sizes(p, rast; window_ranges),
    window_indices=_window_indices(p, rast; window_ranges),
    timed=false,
)
    # Test outputs just return the inputs after window masking 
    if test_windows
        output_stacks = map(window_indices) do i
            _get_window_with_zeroed_buffer(p, rast, window_ranges[i])
        end
        return if mosaic_return
            Rasters.mosaic(sum, collect(skipmissing(output_stacks)); 
                to=rast, missingval=0.0, verbose
            )
        else
            output_stacks
        end
    end

    n = max(length(window_indices), p.threaded ? Threads.nthreads() : 1)
    ch = Channel{NamedTuple}(n)
    for _ in 1:n
        put!(ch, (;))
    end

    sorted_indices = last.(sort!(prod.(window_sizes[window_indices]) .=> window_indices; rev=true))
    verbose && @show sorted_indices
    # Set up channels for threading
    # ch = workspace.channel
    # Define empty outputs
    output_stacks = Vector{RasterStack}(undef, length(sorted_indices))
    # Define a runner for threaded/non-threaded operation
    function run(i, iw)
        # Get a window range
        rs = window_ranges[iw]
        verbose && println("Running job $iw on ranges $rs and thread $(Threads.threadid())")
        # verbose && println("Solving window $i $rs ")
        rast_window = _get_window_with_zeroed_buffer(p, rast, rs)
        # Initialise the window using stored memory
        verbose && println("Getting workspace from channel...")
        workspace = take!(ch)
        verbose && println("Initialising window from size $(size(rast_window)), from ranges $rs...")
        workspace = init!(workspace, p.problem, rast_window; verbose) 
        # Solve for the window
        verbose && println("Solving window $rs...")
        output_stacks[i] = solve!(workspace, p.problem)
        # Return the workspace to the channel
        put!(ch, workspace)
    end
    window_elapsed = Vector{Pair{Float64,Int64}}(undef, length(sorted_indices)) 
    # Run the window problems
    if p.threaded
        Threads.@threads for i in eachindex(sorted_indices)
            iw = sorted_indices[i]
            e = @elapsed run(i, iw)
            window_elapsed[i] = e => iw
        end
    else
        for i in eachindex(sorted_indices)
            iw = sorted_indices[i]
            e = @elapsed run(i, iw)
            window_elapsed[i] = e => iw
        end
    end
    # Maybe mosaic the output
    return if mosaic_return
        t = time()
        result = Rasters.mosaic(sum, output_stacks; to=rast, missingval=0.0, verbose)
        mosaic_elapsed = time() - t
        if timed
            return (; result, window_elapsed, mosaic_elapsed)
        else
            return result
        end
    else
        if timed
            return (; result=output_stacks, mosaic_elapsed)
        else
            return output_stacks
        end
    end
end

init(p::AbstractWindowedProblem, rast::RasterStack; kw...) = init!((;), p, rast; kw...)
function init!(workspace::NamedTuple, p::AbstractWindowedProblem, rast::RasterStack; kw...) 
    # TODO actually allocate 
    n = p.threaded ? Threads.nthreads() : 1
    # workspace = if haskey(workspace, :channel)
    #     workspace
    # else
    #     channel = Channel{NamedTuple}(n)
    #     for _ in 1:n
    #         put!(channel, (;))
    #     end
    #     (; channel)
    # end
    return workspace
end

function _max_window_problem_size(p::AbstractWindowedProblem, rast; kw...)
    sizes = _window_problem_sizes(p, rast; kw...)
    _, i = findmax(prod, sizes)
    return sizes[i]
end

# Calculate the maximum number of source and target values in any window
function _window_problem_sizes(p::AbstractWindowedProblem, rast;
    window_ranges=_window_ranges(p, rast)
)
    # Calculate the maximum number of source and target values in any window
    return map(r -> _problem_size(p, rast, r), window_ranges)
end

_problem_size(p::AbstractProblem, rast) = _problem_size(p, rast, axes(rast))
function _problem_size(p::AbstractProblem, rast, ranges::Tuple)
    source_count = _valid_sources(count, p, rast, ranges)
    target_count = _valid_targets(count, p, rast, ranges)
    return source_count, target_count
end

"""
    BatchProblem(problem::AbstractProblem; buffer, centersize, path, ext)

Combine multiple compute operations into a single object, 
when compute times are long and intermediate storage is needed.

`problem` is usually a [`Problem`](@ref) object or a `WindowedProblem` 
for nested operations.

# Keywords

- `nwindows`: When `problem` is a `WindowedProblem`, the number of windows to use.
    When used, `centersize` and `buffer` are not needed.
- `centersize`: The size of the target square
- `buffer`: The area outside taret square
- `datapath`: The path to store the output rasters.
- `joblistpath`: The path to find the job list.
- `grain`: amount of thinning to apply to the target qualities. `nothing` by default.
    if `2 is used`, the target qualities will be sampled every 2x2 pixels, and should run 4x faster.
- `ext`: The file extension for Rasters.jl to write to. Defaults to `.tif`,
    But can be `.nc` for NetCDF, or most other common extensions.
- `threaded`: Whether to run in parallel. `false` by default. If the problem
    is also threaded at some level it may be faster to set this to `false`.
"""
@kwdef struct BatchProblem{P} <: AbstractWindowedProblem{P}
    problem::P
    buffer::Int
    centersize::Tuple{Int,Int}
    datapath::String
    grain::Union{Nothing,Int} = nothing
    ext::String = ".tif"
    threaded::Bool = false
end
function BatchProblem(problem::Problem; 
    centersize::Union{Int,Tuple{Int,Int}}, kw...
) 
    centersize = centersize isa Tuple{Int,Int} ? centersize : (centersize, centersize)
    BatchProblem(; problem, centersize, kw...)
end
function BatchProblem(problem::WindowedProblem; 
    nwindows=nothing,
    centersize::Union{Nothing,Int,Tuple{Int,Int}}=nothing, 
    buffer::Union{Nothing,Int}=nothing,
    kw...
)
    buffer = if isnothing(buffer) 
        problem.buffer
    else
        buffer == problem.buffer || 
            throw(ArgumentError("BatchProblem buffer must match WindowedProblem buffer. Got $buffer and $(problem.buffer)"))
        buffer
    end
    if isnothing(centersize)
        x = problem.centersize * nwindows
        centersize = x, x
    else
        centersize = centersize isa Tuple{Int,Int} ? centersize : (centersize, centersize)
        map(centersize, ConScape.centersize(problem)) do bcs, wcs 
            rem(bcs, wcs) == 0 ||
                throw(ArgumentError("BatchProblem centersize must be a multiple of WindowedProblem centersize. Got $centersize and $(problem.centersize)"))
        end
        isnothing(nwindows) || throw(ArgumentError("Cannot specify both centersize and nwindows"))
    end
    BatchProblem(; problem, buffer, centersize, kw...)
end

centersize(p::BatchProblem) = p.centersize

function solve(p::BatchProblem, rast::RasterStack; 
    window_indices=_window_indices(p, rast),
    kw...
)
    function run(i) 
        solve(p, rast, i; window_indices, kw...)
    end
    if p.threaded
        Threads.@threads for i in eachindex(window_indices)
            run(i)
        end
    else
        for i in eachindex(window_indices)
            run(i)
        end
    end
end
# Single batch job for running on clusters
function solve(p::BatchProblem, rast::RasterStack, i::Int;
    window_indices=_window_indices(p, rast),
    verbose::Bool=false,
    kw...
)
    # Get the ranges of all jobs
    window_ranges = _window_ranges(p, rast)

    # Get the window range of job i
    rs = window_ranges[window_indices[i]]

    # Get the raster data for job i
    # Just read the whole thing now to reduce reads in overlapping windows
    rast_window = read(_get_window_with_zeroed_buffer(p, rast, rs))

    # Solve for this raster
    output = solve(p.problem, rast_window; kw...)

    # Store the output rasters for this job to disk and return the file path
    return _store(p, output, rs; verbose)
end

function assess(p::AbstractWindowedProblem{<:Problem}, rast::AbstractRasterStack; 
    nthreads=Threads.nthreads(), 
    print=true, 
    kw...
)
    
    # Define the ranges of each window
    window_ranges = _window_ranges(p, rast)

    # Calculate window sizes and allocations
    sizes_and_allocs = map(vec(window_ranges)) do rs
        window_rast = view(rast, rs...)
        sze = _problem_size(p, window_rast)
        allocs = allocations(p.problem, sze; nthreads, kw...)
        sze, allocs
    end

    # Organise stats for each window into vectors
    window_sizes = first.(sizes_and_allocs)
    window_allocations = last.(sizes_and_allocs)
    window_mask = map(s -> prod(s) > 0, window_sizes)
    window_indices = eachindex(window_mask)[window_mask]
    
    # Caclulate allocations, with threading context
    max_allocations = if p.threaded
        # Take the top nthreads allocations 
        # Each thread will need to allocate its own workspace
        sum(sort(window_allocations)[1:min(end, nthreads)])
    else
        # One maximum workspace is allocated and reused
        maximum(window_allocations; init=0)
    end

    # Calculate global stats
    njobs = count(window_mask)
    shape = size(window_ranges)

    return (;
        shape,
        njobs,
        max_allocations,
        window_allocations,
        window_sizes,
        window_mask,
        window_indices,
    )
end
function assess(
    p::AbstractWindowedProblem{<:AbstractWindowedProblem}, 
    rast::AbstractRasterStack; 
    nthreads=Threads.nthreads(),
    print=true,
    verbose=false,
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
    assessments = Vector{Any}(undef, length(window_ranges))
    # Run assessments threaded as they can take a long time for large rasters
    Threads.@threads for i in eachindex(vec(window_ranges))
        rs = window_ranges[i]
        verbose && println("Assessing batch: $i, $rs")
        verbose && println("Retrieving raster from channel...")
        window_rast = take!(channel)
        verbose && println("Copy raster data")
        window_rast = open(rast) do o
            if map(length, rs) == size(window_rast)
                _get_window_with_zeroed_buffer!(window_rast, p, o, rs)
            else
                _get_window_with_zeroed_buffer(getindex, p, o, rs)
            end
        end
        verbose && println("Skipping NaN only rasters...")
        nvalid = count(_isvalid, window_rast.target_qualities)
        assessments[i] = if nvalid > 0
            verbose && println("  nvalid: $nvalid")
            assess(p.problem, window_rast; nthreads, print=false, kw...)
        else
            verbose && println("  No targets found")
            (; 
                shape=(0, 0),
                njobs=0,
                max_allocations=0,
                window_allocations=Int[],
                window_sizes=Tuple{Int,Int}[],
                window_mask=Bool[],
                window_indices=Int[],
            )
        end
        put!(channel, window_rast)
    end

    # Get vectors of vectors from inner problem
    inner_window_allocations = map(a -> a.window_allocations, assessments)
    inner_window_sizes = map(a -> a.window_sizes, assessments)
    inner_window_masks = map(a -> a.window_mask, assessments)
    inner_window_indices = map(a -> a.window_indices, assessments)
    inner_window_counts = map(length, inner_window_sizes)
    inner_window_jobs = map(a -> a.njobs, assessments)

    # Get outer problem vectors
    window_mask = map(any, inner_window_masks)
    window_indices = eachindex(vec(window_mask))[window_mask]

    # Calculate global stats
    max_allocations = if p.threaded
        sum(sort(inner_allocations)[1:min(end, nthreads)])
    else
        maximum(a -> maximum(a; init=0), inner_window_allocations; init=0)
    end
    njobs = count(window_mask)
    max_windows = maximum(inner_window_counts)
    shape = size(window_ranges)

    fields = (; 
        shape,
        njobs, 
        max_windows, 
        max_allocations, 
        window_indices,
        window_mask,
        inner_window_jobs, 
        inner_window_allocations, 
        inner_window_counts, 
        inner_window_sizes,
        inner_window_indices,
        inner_window_masks,
    )

    print && display(pairs(fields))

    return fields
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(p::BatchProblem; to, lazy=true, missingval=0.0, kw...)
    ranges = _window_ranges(p, to)
    paths = [_window_path(p, rs) for rs in ranges]
    stacks = [RasterStack(path; lazy) for path in paths if isdir(path)]

    return Rasters.mosaic(sum, stacks; missingval, to, kw...)
end

function _store(p::BatchProblem, output::RasterStack{K}, ranges; kw...) where K
    dir = mkpath(_window_path(p, ranges))
    return Rasters.write(joinpath(dir, ""), output; 
        ext=p.ext, force=true, verbose=false, kw...
    )
end

function _window_path(p, ranges::Tuple)
    corners = map(first, ranges)
    window_dirname =  "window_" * join(corners, '_')
    return joinpath(p.datapath, window_dirname)
end


### Shared utilities

function _window_indices(p, rast;
    window_ranges=_window_ranges(p, rast),
    window_sizes=_window_sizes(p, rast; window_ranges)
)
    # Get the Bool mask of needed windows
    mask = prod.(window_sizes) .> 0
    # Get the Int indices of the needed windows
    return eachindex(mask)[vec(mask)]
end

function _window_sizes(p, rast::RasterStack; window_ranges=_window_ranges(p, rast))
    map(window_ranges) do rs
        window_rast = view(rast, rs...)
        _problem_size(p, window_rast)
    end
end

function _window_ranges(p::Union{BatchProblem,WindowedProblem}, rast::AbstractRasterStack)
    size = Base.size(rast)
    centersize = ConScape.centersize(p)
    buffer = ConScape.buffer(p)
    windowsize = 2buffer .+ centersize
    cs1, cs2 = centersize
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:cs1:end-2buffer, begin:cs2:end-2buffer]
    # Create an iterator of ranges for retreiving each window
    return [map((i, s, ws) -> i:min(s, i + ws-1), Tuple(c), size, windowsize) for c in corners]
end

# _get_window_with_zeroed_buffer(dest, p, rast, axes(rast))
    
function _get_window_with_zeroed_buffer!(dest, p::AbstractWindowedProblem, rast::RasterStack, rs)
    window = view(rast, rs...)
    dest = rebuild(dest; dims=dims(window))
    # @show typeof(parent(parent(dest.qualities))) typeof(parent(parent(window.qualities)))
    # error()
    parent(parent(dest.qualities)) .= parent(parent(window.qualities))
    parent(parent(dest.affinities)) .= parent(parent(window.affinities))
    return _with_sparse_targets(p, window, dest)
end
_get_window_with_zeroed_buffer(p::AbstractWindowedProblem, args...) = 
    _get_window_with_zeroed_buffer(view, p, args...)
_get_window_with_zeroed_buffer(f::Function , p::AbstractWindowedProblem, rast::RasterStack) = 
    _get_window_with_zeroed_buffer(f, p, rast, axes(rast))
function _get_window_with_zeroed_buffer(f::Function, p::AbstractWindowedProblem, rast::RasterStack, rs)
    window = f(rast, rs...)
    return _with_sparse_targets(p, window, window)
end

function _with_sparse_targets(p, source, dest)
    b = buffer(p)
    tq = source.target_qualities
    tq_sparse = spzeros(eltype(tq), size(tq))
    center_ranges = map(s -> b:s-b, size(tq))
    tq_sparse[center_ranges...] = tq[center_ranges...]
    if !isnothing(grain(p))
        tq_sparse = coarse_graining(tq_sparse, grain(p))
    end

    return merge(dest, (; target_qualities=rebuild(tq; data=tq_sparse)))
end

# Apply function `f` to the validity (Bool) of each window. Empty windows are false. 
# `any` `count` or `map`(for the Vector{Bool}) are useful functions for f
_valid_sources(f, p, rast::AbstractRasterStack) =
    _valid_sources(f, p, rast, axes(rast))
function _valid_sources(f, p, rast::AbstractRasterStack, source_ranges::Tuple)
    # Get a window view
    window = view(rast.qualities, source_ranges...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    return f(_isvalid.(window))
end
function _valid_targets(
    f, p, rast::AbstractRasterStack, source_ranges::Tuple
)
    # Get the range of the target vaues
    b = buffer(p)
    target_ranges = map(source_ranges) do r
        r[b+1:end-b]
    end
    # Get a window view
    window = view(rast.target_qualities, target_ranges...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    return f(_isvalid.(window))
end

_isvalid(x) = !isnan(x) && x > zero(x)

_resolution(rast) = abs(step(lookup(rast, X)))
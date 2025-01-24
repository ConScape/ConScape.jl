# This file is a work in progress...

abstract type AbstractWindowedProblem end

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
@kwdef struct WindowedProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    centersize::Int
    buffer::Int
    threaded::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

function solve(p::WindowedProblem, rast::RasterStack; 
    test_windows=false,
    verbose=false,
    mosaic_return=true
)
    window_ranges = collect(_window_ranges(p, rast))
    # Test outputs just return the inputs after window masking 
    if test_windows
        output_stacks = map(eachindex(window_ranges)) do i
            _mask_target_qualities_overlap!(rast, window_ranges[i], p)
        end
        return if mosaic_return
            Rasters.mosaic(sum, collect(skipmissing(output_stacks)); 
                to=rast, missingval=NaN
            )
        else
            output_stacks
        end
    end

    # Set up channels for threading
    n = p.threaded ? Threads.nthreads() : 1
    ch = Channel{NamedTuple}(n)
    for _ in 1:n
        put!(ch, (;))
    end
    # Define empty outputs
    output_stacks = Vector{RasterStack}(undef, length(window_ranges))
    # Define a runner for threaded/non-threaded operation
    function run(i)
        # Get a window range
        rs = window_ranges[i]
        # verbose && println("Solving window $i $rs ")
        rast_window = _mask_target_qualities_overlap!(rast, rs, p)
        # Initialise the window using stored memory
        workspace = init!(take!(ch), p.problem, rast_window) 
        # Solve for the window
        output_stacks[i] = solve(p.problem, workspace)
        # Return the workspace to the channel
        put!(ch, workspace)
    end
    # Run the window problems
    if p.threaded
        Threads.@threads :greedy for i in eachindex(window_ranges)
            run(i)
        end
    else
        for i in eachindex(window_ranges)
            run(i)
        end
    end
    # Maybe mosaic the output
    return if mosaic_return
        Rasters.mosaic(sum, output_stacks; to=rast, missingval=NaN)
    else
        output_stacks
    end
end

function allocations(p::WindowedProblem, rast::AbstractRasterStack; 
    nthreads=Threads.nthreads(), kw...
)
    range_tuples = _window_ranges(p, rast)
    if p.threaded
        return sum(range_tuples[1:min(end, nthreads)]) do rs
            allocations(p.problem, rast[rs...]; nthreads, kw...)
        end
    else
        return allocations(p.problem, rast[first(range_tuples)...]; nthreads, kw...)
    end
end

function _window_ranges(p, rast)
    window_ranges = collect(_window_ranges(p, rast))
    # We need at least one window
    length(window_ranges) > 0 || throw(ArgumentError("No tiles selected, use a smaller overlap or larger radius"))
    # Get a bitmask of valid windows (not all zeros or NaNs)
    valid_window_mask = _valid_window_mask(p, rast, window_ranges)
    # We only use valid windows
    used_ranges = ranges[valid_window_mask]
    # Sort by size so we can the largest windows first. 
    # This should make threading slightly more efficient as the last tasks 
    # (when some threads are idle) will be the fastest ones.
    # It also allocates the largest arrays first so we can reuse them for the 
    # smaller ones without moving the memory
    sorted_ranges = collect(last.(sort!(map(rs -> prod(_size(p, rast, rs)) => rs, used_ranges))))

    return sorted_ranges
end


function _max_window_size(p::WindowedProblem, rast)
    # TODO make this work nested
    rs = _window_ranges(p, rast)
    # Calculate the maximum number of source and target values in any window
    sizes = map(x -> _size(p, rast, x), rs)
    _, i = findmax(prod, sizes)
    return sizes[i]
end

function _size(p::WindowedProblem, rast, ranges::Tuple)
    source_count = _valid_sources(count, p, rast, ranges)
    target_count = _valid_targets(count, p, rast, ranges)
    return (source_count, target_count)
end

"""
    BatchProblem(problem::AbstractProblem; radius, overlap, path, ext)

Combine multiple compute operations into a single object, 
when compute times are long and intermediate storage is needed.

`problem` is usually a [`Problem`](@ref) object or a `WindowedProblem` 
for nested operations.

# Keywords

- `radius`: The radius of the window - 2radius + 1 is the diameter.
- `overlap`: The overlap between adjacent windows.
- `path`: The path to store the output rasters.
- `ext`: The file extension for Rasters.jl to write to. Defaults to `.tif`,
    But can be `.nc` for NetCDF or most other common extensions.
- `threaded`: Whether to run in parallel. `false` by default
"""
@kwdef struct BatchProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    centersize::Int
    buffer::Int
    datapath::String
    joblistpath::String
    grain::Union{Nothing,Int} = nothing
    ext::String = ".tif"
    threaded::Bool = false
end
BatchProblem(problem; kw...) =  BatchProblem(; problem, kw...)

function solve(p::BatchProblem, rast::RasterStack;
    verbose=false,
)
    ch = Channel{NamedTuple}()
    for _ in 1:Threads.nthreads()
        put!(ch, (;))
    end
    ranges = collect(_window_ranges(p, rast))
    mask = _valid_window_mask(p, rast, ranges)
    used_ranges = ranges[mask]
    function run(i) 
        rs = used_ranges[i]
        verbose && println("Solving window $i $rs ")
        rast_window = _mask_target_qualities_overlap!(rast, rs, p)
        storage = take!(ch)
        workspace = if isnothing(storage)
            init!(storage, p, rast_window)
        end
        output = solve(p.problem, workspace)
        put!(ch, workspace)
        _store(p, output, rs)
    end
    if p.threaded
        Threads.@threads for i in eachindex(used_ranges)
            run(i) 
        end
    else
        for i in eachindex(used_ranges)
            run(i) 
        end
    end
end
# Single batch job for running on clusters
function solve(p::BatchProblem, rast::RasterStack, i::Int;
    verbose=false,
)
    # Indices i are contiguous so we need to spread them
    # accross the actual tiles that need to be done

    # Get all the tile ranges
    ranges = collect(_window_ranges(p, rast))
    # Get the Bool mask of needed windows
    mask = _valid_window_mask(p, rast, ranges)
    # Get the Int indices of the needed windows
    tile_inds = eachindex(mask)[vec(mask)]
    # Get the current window for this job
    rs = ranges[tile_inds[i]]
    # Get the ranges of the window for this job
    rast_window = _mask_target_qualities_overlap!(rast, rs, p)
    # Maybe thin the target qualities
    if !isnothing(p.grain)
        rast_window = ConScape.coarse_graining(rast_window, p.grain)
    end
    output = solve(p.problem, rast[rs...])
    # Store the output rasters for this job to disk
    filename = _store(p, output, rs)
    return filename
end

"""
    count_batches(p::BatchProblem, rast::RasterStack)

Count the number of batch jobs that would need to be run.

A Slurm array job would then be specified "0-(N-1)"

Returns an `Int`.
"""
function count_batches(p::BatchProblem, rast::RasterStack)
    ranges = _window_ranges(p, rast)
    mask = _valid_window_mask(p, rast, ranges)
    return count(mask)
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(p::BatchProblem; 
    to, lazy=false, filename=nothing, missingval=NaN, kw...
)
    ranges = _window_ranges(p, to)
    mask = _valid_window_mask(p, to, ranges)
    paths = [_window_path(p, rs) for (rs, m) in zip(ranges, mask) if m]
    stacks = [RasterStack(path; lazy, name) for path in paths if isdir(path)]

    return Rasters.mosaic(sum, stacks; to, filename, missingval, kw...)
end

function _store(p::BatchProblem, output::RasterStack{K}, ranges) where K
    path = mkpath(_window_path(p, ranges))
    return Rasters.write(joinpath(path, ""), output; 
        ext=p.ext, verbose=false, force=true
    )
end

function _window_path(p, ranges)
    corners = map(first, ranges)
    window_dirname =  "window_" * join(corners, '_')
    return joinpath(p.path, window_dirname)
end


### Shared utilities

_window_ranges(p::Union{BatchProblem,WindowedProblem}, rast::AbstractRasterStack) = 
    _window_ranges(size(rast), p.centersize, p.buffer)
function _window_ranges(size::Tuple{Int,Int}, centersize::Int, buffer::Int)
    windowsize = 2buffer + centersize
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:centersize:end-windowsize, begin:centersize:end-windowsize]
    # Create an iterator of ranges for retreiving each window
    return (map((i, sz) -> i:min(sz, i + windowsize-1), Tuple(c), size) for c in corners)
end

# Create a mask to skip tiles that have no target cells
_valid_window_mask(p, ::Nothing, ranges) = nothing
_valid_window_mask(p, rast::AbstractRasterStack, ranges) =
    map(r -> _valid_targets(any, p, rast, r), ranges)

function _mask_target_qualities_overlap!(rast, rs, p)
    b = p.buffer
    fill = zero(eltype(rast.target_qualities))
    dest = rast[rs...]
    dest.target_qualities[begin:min(begin+b-1, end), :] .= fill
    dest.target_qualities[:, begin:min(begin+b-1, end)] .= fill
    dest.target_qualities[max(end-b+1, begin):end, :] .= fill
    dest.target_qualities[:, max(end-b+1, begin):end] .= fill
    return dest
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
    f(x -> !isnan(x) && x > zero(x), window)
end
function _valid_targets(f, p, rast::AbstractRasterStack, source_ranges::Tuple)
    # Get the range of the target vaues
    o = overlap(p)
    target_ranges = map(source_ranges) do r
        r[o+1:end-o]
    end
    # Get a window view
    window = view(rast.target_qualities, target_ranges...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    f(x -> !isnan(x) && x > zero(x), window)
end

_resolution(rast) = abs(step(lookup(rast, X)))
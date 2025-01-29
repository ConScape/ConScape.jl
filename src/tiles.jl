# This file is a work in progress...

abstract type AbstractWindowedProblem <: AbstractProblem end

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
@kwdef struct WindowedProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    centersize::Int
    buffer::Int
    threaded::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

function solve(p::WindowedProblem, rast::RasterStack; 
    test_windows::Bool=false,
    verbose::Bool=false,
    mosaic_return::Bool=true
)
    window_ranges = _window_ranges(p, rast)
    window_indices = _window_indices(p, rast; window_ranges)
    # Test outputs just return the inputs after window masking 
    if test_windows
        output_stacks = map(window_indices) do i
            _get_window_with_zeroed_buffer(rast, window_ranges[i], p)
        end
        return if mosaic_return
            Rasters.mosaic(sum, collect(skipmissing(output_stacks)); 
                to=rast, missingval=0.0, verbose
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
    output_stacks = Vector{RasterStack}(undef, length(window_indices))
    # Define a runner for threaded/non-threaded operation
    function run(i, ir)
        # Get a window range
        rs = window_ranges[ir]
        # verbose && println("Solving window $i $rs ")
        rast_window = _get_window_with_zeroed_buffer(rast, rs, p)
        # Initialise the window using stored memory
        workspace = init!(take!(ch), p.problem, rast_window) 
        # Solve for the window
        output_stacks[i] = solve!(workspace, p.problem)
        # Return the workspace to the channel
        put!(ch, workspace)
    end
    # Run the window problems
    if p.threaded
        Threads.@threads :greedy for (i, ir) in enumerate(window_indices)
            run(i, ir)
        end
    else
        for (i, ir) in enumerate(window_indices)
            run(i, ir)
        end
    end
    # Maybe mosaic the output
    return if mosaic_return
        Rasters.mosaic(sum, output_stacks; to=rast, missingval=0.0, verbose)
    else
        output_stacks
    end
end

# sorted_ranges = collect(last.(sort!(map(rs -> prod(_size(p, rast, rs)) => rs, used_ranges))))

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
@kwdef struct BatchProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    buffer::Int
    centersize::Tuple{Int,Int}
    datapath::String
    joblistpath::Union{String,Nothing}=nothing
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

function solve(p::BatchProblem, rast::RasterStack; kw...)
    window_indices = _window_indices(p, rast)
    function run(i) 
        solve(p, rast, i; window_indices, kw...)
    end
    if p.threaded
        Threads.@threads :greedy for i in eachindex(window_indices)
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
    window_indices::Bool=nothing, 
    verbose::Bool=false, kw...
)
    # Indices i are contiguous so we need to spread them accross the actual tiles
    # that need to be done by first calculating or retrieving `window_indices`.
    # Manual calculateion is best avoided when it means reading 10gb over a network.
    window_ranges = _window_ranges(p, rast)
    if isnothing(window_indices)
        window_indices = if isnothing(p.joblistpath)
            _window_indices(p, rast)
        else
            _read_joblist(p)
        end
    end

    # Job i
    rs = window_ranges[window_indices[i]]
    # Get the current window for this job
    rast_window = _get_window_with_zeroed_buffer(rast, rs, p)
    output = solve(p.problem, rast_window; kw...)
    # Store the output rasters for this job to disk
    if !ismissing(output)
        _store(p, output, rs; verbose)
    end
    return nothing
end

function assess(p::BatchProblem, rast::RasterStack)
    window_indices = _window_indices(p, rast)
    _write_joblist(p; window_indices)
    a = allocations(p, rast)
    return (; max_allocations=a, njobs=length(window_indices))
end

grain(p::BatchProblem) = p.grain

centersize(p::WindowedProblem) = p.centersize, p.centersize
centersize(p::BatchProblem) = p.centersize

### Batch utilities

function _read_joblist(p::BatchProblem)
    # Read indices from the joblist file. This is generated in `assess`
    isfile(p.joblistpath) || throw(ArgumentError("joblistpath $(p.joblistpath) does not exist"))
    return parse.(Int, readlines(p.joblistpath))
end

function _write_joblist(p::BatchProblem; window_indices)
    if !isnothing(p.joblistpath)
        open(p.joblistpath, "w") do io
            for i in window_indices 
                println(io, i)
            end
        end
    end
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(p::BatchProblem; to, missingval=0.0, kw...)
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
    window_ranges=_window_ranges(p, rast)
)
    # Get the Bool mask of needed windows
    mask = _valid_window_mask(p, rast, window_ranges)
    # Get the Int indices of the needed windows
    return eachindex(mask)[vec(mask)]
end

function _window_ranges(p::Union{BatchProblem,WindowedProblem}, rast::AbstractRasterStack)
    size = Base.size(rast)
    centersize = ConScape.centersize(p)
    buffer = ConScape.buffer(p)
    ws1, ws2 = windowsize = 2buffer .+ centersize
    cs1, cs2 = centersize
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:cs1:end, begin:cs2:end]
    # Create an iterator of ranges for retreiving each window
    return [map((i, s, ws) -> i:min(s, i + ws-1), Tuple(c), size, windowsize) for c in corners]
end

# Create a mask to skip tiles that have no target cells
_valid_window_mask(p, ::Nothing, ranges) = nothing
_valid_window_mask(p, rast::AbstractRasterStack, ranges) =
    map(r -> _valid_targets(any, p, rast, r), ranges)

function _get_window_with_zeroed_buffer(rast, rs, p::AbstractWindowedProblem)
    b = buffer(p)
    fill = zero(eltype(rast.target_qualities))
    dest = if isnothing(grain(p))
        rast[rs...]
    else
        coarse_graining(view(rast, rs), grain(p))
    end
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
    f(x -> !isnan(x) && x > zero(x), window)
end

_resolution(rast) = abs(step(lookup(rast, X)))
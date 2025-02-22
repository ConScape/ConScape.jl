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

# function Base.show(io, mime::MIME"text/plain", p::WindowedProblem)
#     println(io, typeof(p))
#     println(io, "centersize: ", p.centersize)
#     println(io, "buffer:     ", p.buffer)
#     println(io, "threaded:   ", p.threaded)
#     println(io, "problem:    ")
#     show(io, mime, p.problem)
# end

centersize(p::WindowedProblem) = p.centersize, p.centersize
isthreaded(p::WindowedProblem) = p.threaded

function solve(p::WindowedProblem, rast::RasterStack;
    verbose=false, test_windows=false, mosaic_return=true, timed=false, kw...
)
    solve!(init(p, rast; verbose, kw...), p;
        verbose, test_windows, mosaic_return, timed
    )
end
solve!(workspace::Missing, p::WindowedProblem; kw...) = missing
function solve!(workspace::NamedTuple, p::WindowedProblem;
    test_windows::Bool=false,
    mosaic_return::Bool=true,
    timed=false,
    verbose::Bool=false,
)
    (; rast, window_workspaces, window_ranges, selected_window_indices, sorted_indices) = workspace
    # Test outputs just return the inputs after window masking 
    if test_windows
        output_stacks = map(selected_window_indices) do i
            _get_window_with_zeroed_buffer(view, p, rast, window_ranges[i])
        end
        return if mosaic_return
            Rasters.mosaic(sum, collect(skipmissing(output_stacks));
                to=rast, missingval=0.0, verbose
            )
        else
            output_stacks
        end
   end

    ch = Channel{NamedTuple}(length(window_workspaces))
    for ws in window_workspaces
        put!(ch, ws)
    end
    # Set up channels for threading
    # Define empty outputs
    output_stacks = Vector{Union{RasterStack,Missing}}(undef, length(sorted_indices))
    # Define a runner for threaded/non-threaded operation
    function run(i, iw)
        # Get a window range
        window = window_ranges[iw]
        verbose && println("Running job $iw on ranges $window and thread $(Threads.threadid())")
        # verbose && println("Solving window $i $window ")
        window_rast = _get_window_with_zeroed_buffer(view, p, rast, window)
        # Initialise the window using stored memory
        verbose && println("Getting workspace from channel...")
        workspace = take!(ch)
        verbose && println("Initialising window from size $(size(window_rast)), from ranges $window...")
        workspace_initialised = init!(workspace, p.problem, window_rast; verbose)
        # Solve for the window
        verbose && println("Solving window $window...")
        grid = workspace_initialised.grid
        output_stacks[i] = if prod(target_size(grid)) > 0
            solve!(workspace_initialised, p.problem)
        else
            missing
        end
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
        non_missing_output = collect(skipmissing(output_stacks))
        if length(non_missing_output) > 0
            result = Rasters.mosaic(sum, non_missing_output; to=rast, missingval=0.0, verbose)
            mosaic_elapsed = time() - t
            if timed
                (; result, window_elapsed, mosaic_elapsed)
            else
                result
            end
        else
            missing
        end
    else
        if timed
            (; result=output_stacks, window_elapsed)
        else
            output_stacks
        end
    end
end


init(p::WindowedProblem, rast::RasterStack; kw...) = init!((;), p, rast; kw...)
function init!(workspace::NamedTuple, p::WindowedProblem, rast::RasterStack;
    window_ranges=_window_ranges(p, rast),
    grid_sizes=nothing,
    selected_window_indices=nothing,
    verbose=true,
)
    grid_sizes = isnothing(grid_sizes) ? _estimate_grid_sizes(p, rast; window_ranges) : grid_sizes
    selected_window_indices = isnothing(selected_window_indices) ? _select_indices(p, rast; window_ranges, grid_sizes) : selected_window_indices
    sorted_indices = last.(sort!(prod.(grid_sizes[selected_window_indices]) .=> selected_window_indices; rev=true))
    length(sorted_indices) > 0 || return missing
   
    n = min(length(selected_window_indices), p.threaded ? Threads.nthreads() : 1)
    window_workspaces = Vector{NamedTuple}(undef, n)
    if haskey(workspace, :window_workspaces)
        Threads.@threads for i in 1:n
            window_workspaces[i] = init!(workspace.window_workspaces[i], p.problem; verbose)
        end
    else
        largest_rast = _get_window_with_zeroed_buffer(view, p, rast, window_ranges[first(sorted_indices)])
        Threads.@threads for i in 1:n
            window_workspaces[i] = init(p.problem, largest_rast; verbose)
        end
    end
    return (; rast, window_workspaces, grid_sizes, window_ranges, selected_window_indices, sorted_indices)
end

function _max_estimated_grid_size(p::AbstractWindowedProblem, rast; kw...)
    sizes = _estimate_grid_sizes(p, rast; kw...)
    _, i = findmax(prod, sizes)
    return sizes[i]
end


# Calculate the maximum number of source and target values in any window
function _estimate_grid_sizes(p::AbstractWindowedProblem, rast;
    window_ranges=_window_ranges(p, rast)
)
    # Calculate the maximum number of source and target values in any window
    return map(r -> _estimate_grid_size(p, rast, r), window_ranges)
end

# This function extimates problem size without actually constructing grids.
# It cant be too small, but may be too large
_estimate_grid_size(p::AbstractProblem, rast) = _estimate_grid_size(p, rast, axes(rast))
function _estimate_grid_size(p::AbstractProblem, rast, ranges::Tuple)
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

# function Base.show(io, mime::MIME"text/plain", p::BatchProblem)
#     println(io, typeof(p))
#     println(io, "centersize: ", p.centersize)
#     println(io, "buffer:     ", p.buffer)
#     println(io, "datapath:   ", p.datapath)
#     println(io, "ext:        ", p.ext)
#     println(io, "grain:      ", p.grain)
#     println(io, "problem:    ")
#     show(io, mime, p.problem)
# end

centersize(p::BatchProblem) = p.centersize

function solve(p::BatchProblem, rast::RasterStack;
    batch_indices=_select_indices(p, rast), kw...
)
    for i in eachindex(batch_indices)
        solve(p, rast, i; batch_indices, kw...)
    end
end
function solve(p::BatchProblem, rast::RasterStack, i; verbose=false, kw...)
    solve!(init(p, rast, i; verbose, kw...), p; verbose, kw...)
end
# Single batch job for running on clusters
function solve!(ws::NamedTuple, p::BatchProblem; verbose=false, kw...)
    output = solve!(ws.workspace, p.problem; verbose) # Store the output rasters for this job to disk and return the fiee path
    return if ismissing(output) 
        missing
    else
        non_missing_output = collect(skipmissing(output))
        @show length(non_missing_output)
        if length(non_missing_output) > 0
            _store(p, non_missing_output, ws.batch_ranges; verbose)
        else
            missing
        end
    end
end

function init(p::BatchProblem{<:WindowedProblem}, rast::RasterStack, i::Int;
    batch_ranges=_window_ranges(p, rast),
    batch_indices=(println("Calculating batch indices, pass `batch_indices` to skip... "); _select_indices(p, rast; window_ranges=batch_ranges)),
    window_indices=nothing,
    grid_sizes=nothing,
    verbose=false,
)
    # Get the raster data for job i
    ranges = batch_ranges[batch_indices[i]]
    verbose && @show ranges
    # We want to materialise the raster, and we don't need sparse targets
    batch_rast = rast[ranges...]

    window_ranges = _window_ranges(p, batch_rast)
    grid_sizes = isnothing(grid_sizes) ? _estimate_grid_sizes(p, batch_rast) : grid_sizes[i]
    selected_window_indices = isnothing(window_indices) ? _select_indices(p, batch_rast; window_ranges, grid_sizes) : window_indices[i]
    workspace = init(p.problem, batch_rast; verbose, grid_sizes, selected_window_indices, window_ranges)
    return (; workspace, batch=i, batch_ranges)
end
function init(p::BatchProblem{<:Problem}, rast::RasterStack, i::Int;
    batch_ranges=_window_ranges(p, rast),
    batch_indices=(println("Calculating batch indices, pass `batch_indices` to skip... "); _select_indices(p, rast; window_ranges=batch_ranges)),
    verbose=false,
)
    # Get the raster data for job i
    ranges = batch_ranges[batch_indices[i]]
    verbose && @show ranges
    # Materialise the window, but with sparse targets
    batch_rast = _get_window_with_zeroed_buffer(getindex, p, rast, ranges)
    worskpace = init(p.problem, batch_rast; verbose)
    return (; workspace, batch=i, batch_ranges)
end
function Rasters.mosaic(p::BatchProblem; to, lazy=true, missingval=0.0, kw...)
    paths = batch_paths(p, to)
    stacks = [RasterStack(path; lazy) for path in paths if isdir(path)]
    return Rasters.mosaic(sum, stacks; missingval, to, kw...)
end

function _store(p::BatchProblem, output::RasterStack{K}, ranges; kw...) where {K}
    dir = mkpath(_batch_path(p, ranges))
    return Rasters.write(joinpath(dir, ""), output;
        ext=p.ext, force=true, verbose=false, kw...
    )
end

batch_paths(p, x::Union{RasterStack,Tuple}; batch_ranges=_window_ranges(p, x)) = 
    [_batch_path(p, rs) for rs in batch_ranges]

function _batch_path(p, ranges::Tuple)
    corners = map(first, ranges)
    dirname = "batch_" * join(corners, '_')
    return joinpath(p.datapath, dirname)
end


### Shared utilities

# Select the windows in rast that a likely to have valid targets
# pruning may further remove some windows, but is too expensive to do here
# Running `assess` before solving to do this perfectly.
function _select_indices(p, rast;
    window_ranges=_window_ranges(p, rast),
    grid_sizes=_grid_sizes(p, rast; window_ranges)
)
    # Get the Bool mask of needed windows
    mask = prod.(grid_sizes) .> 0
    # Get the Int indices of the needed windows
    return eachindex(mask)[vec(mask)]
end

_window_ranges(p::Union{BatchProblem,WindowedProblem}, rast::AbstractRasterStack) =
    _window_ranges(p::Union{BatchProblem,WindowedProblem}, size(rast))
function _window_ranges(p::Union{BatchProblem,WindowedProblem}, size::Tuple)
    centersize = ConScape.centersize(p)
    buffer = ConScape.buffer(p)
    windowsize = 2buffer .+ centersize
    cs1, cs2 = centersize
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:cs1:end-2buffer, begin:cs2:end-2buffer]
    # Create an iterator of ranges for retreiving each window
    return [map((i, s, ws) -> i:min(s, i + ws - 1), Tuple(c), size, windowsize) for c in corners]
end

function _get_window_with_zeroed_buffer!(dest, p::AbstractWindowedProblem, rast::RasterStack, rs)
    source = view(rast, rs...)
    # Reshape and rebuild to resuse memory
    data = (
        affinities=_reshape(parent(parent(dest.affinities)), size(source)),
        qualities=_reshape(parent(parent(dest.qualities)), size(source)),
        target_qualities=parent(parent(dest.target_qualities)),
    )
    dest = rebuild(dest; data, dims=dims(source))
    # Update values
    dest.qualities .= source.qualities
    dest.affinities .= source.affinities

    return _with_sparse_targets(p, source, dest)
end
_get_window_with_zeroed_buffer(p::AbstractWindowedProblem, args...) =
    _get_window_with_zeroed_buffer(view, p, args...)
_get_window_with_zeroed_buffer(f::Function, p::AbstractWindowedProblem, rast::RasterStack) =
    _get_window_with_zeroed_buffer(f, p, rast, axes(rast))
function _get_window_with_zeroed_buffer(f::Function, p::AbstractWindowedProblem, rast::RasterStack, rs)
    source = f(rast, rs...)
    return _with_sparse_targets(p, source, source)
end

_target_ranges(p, source) = map(s -> buffer(p)+1:s-buffer(p), size(source))

function _with_sparse_targets(p, source, dest)
    tq = source.target_qualities
    tq_sparse = spzeros(eltype(tq), size(tq))
    target_ranges = _target_ranges(p, source)
    tq_sparse[target_ranges...] = tq[target_ranges...]
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
_isvalid(x::Bool) = x

_resolution(rast) = abs(step(lookup(rast, X)))
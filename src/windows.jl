# This file is a work in progress...
abstract type AbstractWindowedProblem{P} <: AbstractProblem end

buffer(p::AbstractWindowedProblem) = p.buffer
buffer(p::AbstractProblem) = 0
grain(::AbstractProblem) = nothing
problem(p::AbstractWindowedProblem) = p.problem
costfunction(p::AbstractWindowedProblem) = costfunction(problem(p))
solver(p::AbstractWindowedProblem) = solver(problem(p))
shape(p::AbstractWindowedProblem) = p.shape

"""
    WindowedProblem(problem::AbstractProblem; kw...)

Combine multiple compute operations into a single object, 
to be run over windowed grids.

## Arguments

`problem`: A [`Problem`](@ref) object.

## Keywords

- `centersize::Int`: The size of one side of the square of target 
    pixels, in the center of the window.
- `buffer`: The maximum number of pixels outside the target square.
- `shape`: whether to make the shape of the window a :square or a :circle.
    using `:circle` will give faster runtime and leave less artifacts.
- `threaded`: Whether to run windows in parallel on separate threads. `false` by default.
- `mosaic_return`: Whether to `mosaic` returned spatial rasters from each window, 
    or return them as a vector. This can be useful for diagnostics. `true` by default.
- `gc`: Whether to run the garbage collector between windows. This may be important in
    restricted memory environments, such as a small node on a SLURM cluster. 
    `true` by default. It may improve performance to set to `false`.
"""
@kwdef struct WindowedProblem{P} <: AbstractWindowedProblem{P}
    problem::P
    centersize::Int
    buffer::Int
    threaded::Bool = false
    shape::Symbol = :circle
    gc::Bool = true 
    test_windows::Bool = false
    mosaic_return::Bool = true
    timed::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

# function Base.show(io::IO, ::MIME"text/plain", p::WindowedProblem)
#     summary(io, p)
#     println(io, "centersize: ", p.centersize)
#     println(io, "buffer:     ", p.buffer)
#     println(io, "threaded:   ", p.threaded)
#     println(io, "gc:         ", p.gc)
# end

centersize(p::WindowedProblem) = p.centersize, p.centersize
isthreaded(p::WindowedProblem) = p.threaded

struct WindowedInit{P,R} <: Initialisation
    problem::P
    rast::R
    sparse_sizes::Vector{Tuple{Int,Int}}
    ranges::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    indices::Vector{Int}
    sorted_indices::Vector{Int}
end

problem(wi::WindowedInit) = wi.problem

function init(p::WindowedProblem, rast::RasterStack; 
    window_ranges=window_ranges(p, rast),
    sparse_sizes=nothing,
    indices=nothing,
    verbose=true,
    kw...
)
    window_ranges = vec(window_ranges)
    # Estimate grid sizes. This is expensive, it us usually passed in from an Assessment
    sparse_sizes = isnothing(sparse_sizes) ? _estimate_sparse_sizes(p, rast; window_ranges) : sparse_sizes |> vec
    @assert length(window_ranges) == length(sparse_sizes)

    # Select and sort indices: we usually only run a subset of windows
    indices = isnothing(indices) ? _select_indices(p, rast; window_ranges, sparse_sizes) : indices
    sorted_indices = last.(sort!(first.(sparse_sizes[indices]) .=> indices; rev=true))

    return WindowedInit(
        p, rast, sparse_sizes, window_ranges, indices, sorted_indices
    )
end
function init(wi::WindowedInit, i::Int; verbose=false)
    ranges = wi.ranges[i]
    verbose && println("Initialising window from ranges $ranges...")
    rast = _get_window_with_zeroed_buffer(wi, ranges)
    init(problem(problem(wi)), rast; verbose)
end

solve(p::WindowedProblem, rast::RasterStack; verbose=false, kw...) =
    solve(init(p, rast; verbose, kw...); verbose)
function solve(window_init::WindowedInit; 
    verbose::Bool=false,
    mosaic_return=problem(window_init).mosaic_return,
    timed=problem(window_init).timed,
)
    (; rast, ranges, indices, sorted_indices) = window_init
    verbose && println("Solving WindowedProblem with $(length(indices)) windows...")
    p = problem(window_init)
    # Test outputs just return the inputs after window masking 
    if p.test_windows
        output_stacks = map(indices) do i
            _get_window_with_zeroed_buffer(p, rast, ranges[i])
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
    # Define a runner for threaded/non-threaded operation
    function run(i, iw)
        verbose && println("Running window $i - $iw on thread $(Threads.threadid())")
        # Initialise the window using stored memory
        multi_grid_init = init(window_init, iw; verbose)
        @assert multi_grid_init isa MultiGridInit
        # Solve for the window
        elapsed = @elapsed begin
            output = solve(multi_grid_init; verbose)
        end
        # Garbage collect for this window
        # Inneficient for few/small windows but
        # often needed for many/large windows
        p.gc && GC.gc()
        return output, elapsed
    end
    # Run the window problems
    out_elapsed = if p.threaded
        fetch.([Threads.@spawn run(i, sorted_indices[i]) for i in eachindex(sorted_indices)])
    else
        [run(i, sorted_indices[i]) for i in eachindex(sorted_indices)]
    end
    output_stacks = first.(out_elapsed)
    window_elapsed = last.(out_elapsed)
    # Maybe mosaic the output
    return if mosaic_return
        t = time()
        non_missing_output = collect(skipmissing(output_stacks))
        if length(non_missing_output) > 0
            result = Rasters.mosaic(sum, non_missing_output; to=rast, missingval=NaN, verbose)
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

"""
    BatchProblem(problem::AbstractProblem; buffer, centersize, path, ext)

Split a large `Problem` into windowed batches, similar to `WindowedProblem`,
but allow launching individual batches separately with a batch id, and stores
them to separate files when finished, rather than returning the finished job.   

`BatchProblem` is useful when compute times are long and intermediate storage is needed,
and is designed for use with SLURM and similar computate clusters.

`problem` can be a [`Problem`](@ref) object or a `WindowedProblem` for nested operations.
Deciding to use a Problem or NestedProblem will depend on the tradeoffs of loading and 
saving raster data for each window area. This may be relatively expensive if the batch windows are
not very large. Due to the ON^2 scaling of connectivity calculations large batches will also 
become expensive using `Problem` directly.

If `problem` is a `WindowedProblem` IO overheads should be negligible in
comparison to the workload of running `solve` for all windows in a batch.

## Keywords

- `nwindows`: When `problem` is a `WindowedProblem`, the number of windows to use.
    When used, `centersize` and `buffer` are not needed.
- `centersize`: The size of the target square.
- `buffer`: The number of pixels outside target square. If using a `WindowedProblem`,
    this is not needed, but if passed it must match.
- `datapath`: The path to store the output rasters.
- `grain`: amount of thinning to apply to the target qualities. `nothing` by default.
    if `2 is used`, the target qualities will be sampled every 2x2 pixels, and should run 4x faster.
- `ext`: The file extension for Rasters.jl to write to. Defaults to `.tif`,
    But can be `.nc` for NetCDF, or most other common extensions.

BatchProblem is designed so that `init`, `init!`, `solve` and `solve!` can all be called on 
`f(p::BatchProblem, rast::RasterStack)` to run all batches, or with a batch number
`f(p::BatchProblem, rast::RasterStack, batch::Int)` to run a single batch.

# Example

Calculating `init` for `BatchProblem` is relatively expensive. Running [`assess`](@ref) 
first is the best option, and is intended to allow assessment of the scale of the 
problem, as it may require hundreds or thousands of CPU hours to complete. 

With this approach, batches can be run with:

```julia
using ConScape, JSON3, MyConScapeApp
batchproblem = MyConScapeApp.define_my_batchproblem()
rast = MyConScapeApp.get_my_rasterstack()
assessment = assess(batchproblem, rast) # Will take a long time
JSON3.write("assessment.json")
```

Noticed we defined our own application package MyConScapeApp. This is a good way to 
share functions like `define_my_batchproblem` accross multiple task launches on a cluster. 
See the ConScape GitHub organisation for working examples of packages like this, ConScapeJobs.

```julia
usign ConScape, Rasters, MyConScapeApp
batch_problem = define_my_batch()
rast = get_my_rasterstack()
assessment = JSON3.read("assessment.json")
batch_id = 1
# And here we pass the assesment and batch id to `solve`
solve(batch_problem, rast, assessment, batch_id)
```

Finally, when all batches have run we can mosaic the results together

```julia
usign ConScape, Rasters, MyConScapeApp
batch_problem = define_my_batch()
rast = get_my_rasterstack()
# Setting `to` to the original raster ensures the output matches it spatially.
mosaic(batch_problem; to=rast, filename="dest_filename.tif")
```
"""
@kwdef struct BatchProblem{P} <: AbstractWindowedProblem{P}
    problem::P
    buffer::Int
    centersize::Tuple{Int,Int}
    shape::Symbol = :circle
    datapath::String
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
    end
   
    BatchProblem(; problem, buffer, centersize, kw...)
end

centersize(problem::BatchProblem) = problem.centersize

# function Base.show(io, ::MIME"text/plain", p::BatchProblem)
#     summary(io, p)
#     println(io, "centersize: ", p.centersize)
#     println(io, "buffer:     ", p.buffer)
#     println(io, "datapath:   ", p.datapath)
#     println(io, "ext:        ", p.ext)
#     println(io, "grain:      ", p.grain)
#     summary(io, problem(p))
# end

@kwdef struct BatchInit{P<:AbstractProblem,R<:RasterStack} <: Initialisation
    problem::P
    rast::R
    batch_ranges::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    batch_indices::Vector{Int}
    window_indices::Union{Nothing,Vector{Vector{Int}}} = nothing
    sparse_sizes::Union{Nothing,Vector{Vector{Tuple{Int,Int}}}} = nothing
end

problem(wi::BatchInit) = wi.problem

solve(problem::BatchProblem, rast::RasterStack, i::Int...; verbose=false, kw...) =
    solve(init(problem, rast; kw...), i...; verbose)

# Initialise BatchProblem to a BatchInit
init(problem::BatchProblem, rast::RasterStack, i::Int; verbose=false, kw...) =
    init(init(problem, rast; verbose, kw...), i; verbose)
function init(problem::BatchProblem, rast::RasterStack; 
    batch_ranges=window_ranges(problem, rast),
    batch_indices=_select_indices(problem, rast; window_ranges=batch_ranges),
    verbose=false, kw...
)
    verbose && println("Initialising batch problem for RasterStack of size $(size(rast)) and $(length(batch_ranges)) batches...")
    return BatchInit(; 
        problem, 
        rast, 
        batch_ranges=vec(batch_ranges), 
        batch_indices=vec(batch_indices),
        kw...
    )
end
# Initialise a job from a BatchInit to WindowedInit for WindowedProblem

function init(bi::BatchInit{<:BatchProblem{<:WindowedProblem}}, i::Int; verbose=false, kw...)
    checkbounds(Bool, bi.batch_indices, i) || 
        throw(ArgumentError("Invalid batch index $i, must be between 1 and $(length(bi.batch_indices))"))

    indices = isnothing(bi.window_indices) ? nothing : bi.window_indices[bi.batch_indices[i]]
    sparse_sizes = isnothing(bi.sparse_sizes) ? nothing : bi.sparse_sizes[bi.batch_indices[i]]
    ranges = bi.batch_ranges[bi.batch_indices[i]]

    verbose && println("Loading raster window for ranges $ranges...")
    rast = bi.rast[ranges...]
    verbose && println("Initialising window problem")
    return init(problem(problem(bi)), rast; verbose, indices, sparse_sizes)
end
# Or to GridInit for Problem
function init(bi::BatchInit{<:BatchProblem{<:Problem}}, i::Int; verbose=false, kw...)
    ranges = bi.batch_ranges[bi.batch_indices[i]]
    problem_rast = _get_window_with_zeroed_buffer(bi, ranges)
    return init(problem(problem(bi)), problem_rast; verbose)
end

# Solve a single batch job (there is no method to solve all jobs)
function solve(bi::BatchInit, i::Int; verbose=false)
    verbose && println("Initialising window memory...")
    inner_init = init(bi, i; verbose)
    ib = bi.batch_indices[i]
    ranges = bi.batch_ranges[ib]
    verbose && println("Running batch $i for window $ib over ranges $ranges...")
    output = solve(inner_init; verbose)
     # Store the output raster/s for this job to disk and return the file path
    return if ismissing(output)
        println("WARNING: Output was empty for job $i at window $ib over ranges $ranges")
        missing
    else
        # Clear out some memory before writing
        GC.gc()
        _write(problem(bi), output, ranges; verbose)
    end
end
# Solve all batches. 
# Not the main intent of `BatchProblem` but here as a convenience.
solve(bi::BatchInit; verbose=false) =
    [solve(bi, i; verbose) for i in eachindex(bi.batch_indices)]

"""
    batch_paths(p::BatchProblem, x::RasterStack)
    batch_paths(p::BatchProblem, size::Tuple{Int,Int})
    batch_paths(p::BatchInit, a::NestedAssessment)
    batch_paths(p::BatchInit)

Get a `Vector` of `String` folder paths for each batch.

If a `BatchProblem` and a `NestedAssessment` from `assess` are used, 
or a `BatchInit` is used directly, the paths will be only those that 
need to run (i.e. that are not empty). Otherwise this will include paths 
for all possible windows, even those that will not run as batch jobs.
"""
batch_paths(p::BatchProblem, x::Union{RasterStack,Tuple{Int,Int}}; batch_ranges=window_ranges(p, x)) = 
    [_batch_path(p, rs) for rs in batch_ranges]
batch_paths(bi::BatchInit) = 
    [_batch_path(problem(bi), rs) for i in bi.batch_indices for rs in bi.batch_ranges[i]]

function _batch_path(p, ranges::Tuple{<:UnitRange{Int},<:UnitRange{Int}})
    corners = map(first, ranges)
    dirname = "batch_" * join(corners, '_')
    return joinpath(p.datapath, dirname)
end

# Write the output to disk with Rasters
function _write(p::BatchProblem, output::RasterStack{K}, ranges::Tuple; verbose, kw...) where {K}
    dir = mkpath(_batch_path(p, ranges))
    verbose && println("Writing finished raster to $dir...")
    return Rasters.write(joinpath(dir, ""), output;
        ext=p.ext, force=true, verbose, kw...
    )
end

### Shared utilities

# Select the windows in rast that a likely to have valid targets
# pruning may further remove some windows, but is too expensive to do here
# Running `assess` before solving to do this perfectly.
function _select_indices(p, rast;
    window_ranges=window_ranges(p, rast),
    sparse_sizes=_estimate_sparse_sizes(p, rast; window_ranges)
)
    # Get the Bool mask of needed windows
    mask = prod.(sparse_sizes) .> 0
    # Get the Int indices of the needed windows
    return eachindex(mask)[vec(mask)]
end

window_ranges(wi::Union{WindowedInit,BatchInit}) = window_ranges(problem(wi), wi.rast)
window_ranges(p::Union{BatchProblem,WindowedProblem}, rast::AbstractRasterStack) =
    window_ranges(p::Union{BatchProblem,WindowedProblem}, size(rast))
function window_ranges(p::Union{BatchProblem,WindowedProblem}, size::Tuple)
    centersize = ConScape.centersize(p)
    buffer = ConScape.buffer(p)
    windowsize = 2buffer .+ centersize
    cs1, cs2 = centersize
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:cs1:end-2buffer, begin:cs2:end-2buffer]
    # Create an iterator of ranges for retreiving each window
    _to_ranges(i, s, ws) = (i:min(s, i + ws - 1))::UnitRange{Int}
    return [map(_to_ranges, Tuple(c), size, windowsize)::NTuple{2} for c in corners]
end

_get_window_with_zeroed_buffer(wi::Union{WindowedInit,BatchInit}, args...; kw...) =
    _get_window_with_zeroed_buffer(problem(wi), wi.rast, args...; kw...)
_get_window_with_zeroed_buffer(p::AbstractWindowedProblem, rast::RasterStack; kw...) =
    _get_window_with_zeroed_buffer(p, rast, axes(rast); kw...)
function _get_window_with_zeroed_buffer(
    p::AbstractWindowedProblem, rast::RasterStack, 
    rs::Tuple{<:AbstractUnitRange,<:AbstractUnitRange};
    shape=shape(p)
)
    window = view(rast, rs...)
    tq = _get_target_qualities(window)
    tq_sparse = spzeros(eltype(tq), size(tq))
    target_ranges = _target_ranges(p, window)
    tq_sparse[target_ranges...] = tq[target_ranges...]
    if !isnothing(grain(p))
        tq_sparse = coarse_graining(tq_sparse, grain(p))
    end

    target_qualities = rebuild(tq; data=tq_sparse)
    source_qualities = modify(Array, _get_source_qualities(window))
    
    # Handle :circle shaped buffers
    if shape == :circle
        center = CartesianIndex(size(source_qualities) .÷ 2 .+ 1)
        maxdist = buffer(p) + max(centersize(p)...) / 2
        for I in CartesianIndices(source_qualities)
            if _dist_from_center(I, center) >= maxdist
                source_qualities[I] = 0.0
            end
        end
    elseif shape != :square
        error("WindowedProblem shape must be :square or :circle")
    end

    return merge(window, (; source_qualities, target_qualities))
end

function _dist_from_center(point::CartesianIndex, center::CartesianIndex)
    map(Tuple(point), Tuple(center)) do pn, cn
        (pn - cn)^2
    end |> sum |> sqrt
end

_target_ranges(p, source) = map(s -> buffer(p)+1:s-buffer(p), size(source))

# Apply function `f` to the validity (Bool) of each window. Empty windows are false. 
# `any` `count` or `map`(for the Vector{Bool}) are useful functions for f
_valid_sources(f, p, rast::AbstractRasterStack) =
    _valid_sources(f, p, rast, axes(rast))
function _valid_sources(f, p, rast::AbstractRasterStack, source_ranges::Tuple)
    # Get a window view
    window = view(_get_source_qualities(rast), source_ranges...)
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
    window = view(_get_target_qualities(rast), target_ranges...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    return f(_isvalid.(window))
end

_isvalid(x) = !isnan(x) && x > zero(x)
_isvalid(x::Bool) = x

_resolution(rast) = abs(step(lookup(rast, X)))

function Rasters.mosaic(p::BatchProblem, rast::RasterStack; filename=nothing, lazy=true, force=false, progress=true)
    paths = filter(isdir, batch_paths(p, rast))
    isempty(paths) && error("No directories exist to mosaic, have any batches been run?")
    stacks = [RasterStack(path; lazy=true) for path in paths]
    return mosaic(sum, stacks; filename, lazy, force, progress, missingval=NaN)
end
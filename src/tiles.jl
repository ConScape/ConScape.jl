# This file is a work in progress...

abstract type AbstractWindowedProblem end

function allocations(p::AbstractWindowedProblem, sze::Tuple{Int,Int})
end

"""
    WindowedProblem(problem::AbstractProblem; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.

`problem` is usually a [`Problem`](@ref) object but can be any `AbstractProblem`.

# Keywords

- `problem`: The radius of the window.
- `radius`: The radius of the window.
- `overlap`: The overlap between windows.
- `threaded`: Whether to run in parallel. `false` by default
"""
@kwdef struct WindowedProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    radius::Int
    overlap::Int
    threaded::Bool = false
end
WindowedProblem(problem; kw...) = WindowedProblem(; problem, kw...)

function sizeofallocations(p::Problem, sze::Tuple{Int,Int})
    gms = graph_measures(p)
    A_size = sizeofAs(gms, sze)
    Z_size = sizeofZs(gms, sze)
    init_size = sizeofinits(solver(p), gms, sze)

    return_size = sum(map(sizeofreturn, gms, sze))

    return A_size + Z_size + init_size + return_size
end

function solve(p::WindowedProblem, rast::RasterStack; 
    test_windows=false,
    verbose=false,
)
    ranges = collect(_get_window_ranges(p, rast))
    mask = _get_window_mask(rast, ranges)
    output_stacks = Vector{RasterStack}(undef, count(mask))
    used_ranges = ranges[mask]
    if test_windows
        output_stacks = map(eachindex(used_ranges)) do i
            _mask_target_qualities_overlap!(rast, used_ranges[i], p)
        end
        return Rasters.mosaic(sum, output_stacks; to=rast, missingval=NaN)
    end
    function run(i)
        rs = used_ranges[i]
        verbose && println("Solving window $i $rs ")
        rast_window = _mask_target_qualities_overlap!(rast, rs, p)
        output_stacks[i] = solve(p.problem, rast_window)#; workspace)
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
    # Return mosaics of outputs
    return Rasters.mosaic(sum, output_stacks; to=rast, missingval=NaN)
end

# function assess(op::WindowedProblem, g::Grid) 
#     window_assessments = map(_windows(op, g)) do w
#         ca = assess(op.op, w)
#     end
#     maximums = reduce(window_assessments) do acc, a
#         (; totalmem=max(acc.totalmem, a.totalmem),
#            zmax=max(acc.zmax, a.zmax),
#            lumax=max(acc.lumax, a.lumax),
#         )
#     end
#     ComputeAssesment(; op=op.op, maximums..., sums...)
# end


"""
    StoredProblem(problem::AbstractProblem; radius, overlap, path, ext)

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
@kwdef struct StoredProblem <: AbstractWindowedProblem
    problem::AbstractProblem
    radius::Int
    overlap::Int
    path::String
    grain::Union{Nothing,Int} = nothing
    ext::String = ".tif"
    threaded::Bool = false
end
StoredProblem(problem; kw...) =  StoredProblem(; problem, kw...)

function solve(p::StoredProblem, rast::RasterStack;
    verbose=false,
    # workspace=init(p, rast),
)
    ranges = collect(_get_window_ranges(p, rast))
    mask = _get_window_mask(rast, ranges)
    used_ranges = ranges[mask]
    function run(i) 
        rs = used_ranges[i]
        verbose && println("Solving window $i $rs ")
        rast_window = _mask_target_qualities_overlap!(rast, rs, p)
        output = solve(p.problem, rast_window)#; workspace)
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
function solve(p::StoredProblem, rast::RasterStack, i::Int;
    verbose=false,
)
    # Indices i are contiguous so we need to spread them
    # accross the actual tiles that need to be done

    # Get all the tile ranges
    ranges = collect(_get_window_ranges(p, rast))
    # Get the Bool mask of needed windows
    mask = _get_window_mask(rast, ranges)
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
    count_batches(p::StoredProblem, rast::RasterStack)

Count the number of batch jobs that would need to be run.

A Slurm array job would then be specified "0-(N-1)"

Returns an `Int`.
"""
function count_batches(p::StoredProblem, rast::RasterStack)
    ranges = _get_window_ranges(p, rast)
    mask = _get_window_mask(rast, ranges)
    return count(mask)
end

# Mosaic the stored files to a RasterStack
function Rasters.mosaic(p::StoredProblem; 
    to, lazy=false, filename=nothing, missingval=NaN, kw...
)
    ranges = _get_window_ranges(p, to)
    mask = _get_window_mask(to, ranges)
    paths = [_window_path(p, rs) for (rs, m) in zip(ranges, mask) if m]
    stacks = [RasterStack(path; lazy, name) for path in paths if isdir(path)]

    return Rasters.mosaic(sum, stacks; to, filename, missingval, kw...)
end

function _store(p::StoredProblem, output::RasterStack{K}, ranges) where K
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

_get_window_ranges(p::Union{StoredProblem,WindowedProblem}, rast::AbstractRasterStack) = 
    _get_window_ranges(size(rast), p.radius, p.overlap)
function _get_window_ranges(size::Tuple{Int,Int}, r::Int, overlap::Int)
    2r <= overlap && throw(ArgumentError("2 * radius must be larger than overlap"))
    d = 2r
    s = d - overlap # Step between each window corner
    # Define the corners of each window
    corners = CartesianIndices(size)[begin:s:end, begin:s:end]
    # Create an iterator of ranges for retreiving each window
    return (map((i, sz) -> i:min(sz, i + d), Tuple(c), size) for c in corners)
end

_get_window_mask(::Nothing, ranges) = nothing
_get_window_mask(rast::AbstractRasterStack, ranges) =
    _get_window_mask(_get_target(rast), ranges)
function _get_window_mask(target::AbstractRaster, ranges)
    # Create a mask to skip tiles that have no target cells
    map(r -> _has_values(target, r), ranges)
end

function _mask_target_qualities_overlap!(rast, rs, p, last=false)
    o = p.overlap
    fill = zero(eltype(rast.target_qualities))
    dest = rast[rs...]
    dest.target_qualities[max(begin, end-o):end, :] .= fill
    dest.target_qualities[begin:min(end,begin+o), :] .= fill
    dest.target_qualities[:, max(begin, end-o):end] .= fill
    dest.target_qualities[:, begin:min(end, begin+o)] .= fill
    return rast
end

function _has_values(target::AbstractRaster, rs::Tuple{Vararg{AbstractUnitRange}})
    # Get a window view
    window = view(target, rs...)
    # If there are non-NaN cells above zero, keep the window
    # TODO allow users to change this condition?
    any(x -> !isnan(x) && x > zero(x), window)
end

_resolution(rast) = abs(step(lookup(rast, X)))

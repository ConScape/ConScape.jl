"""
    WindowedProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.

"""
@kwdef struct WindowedProblem{P,S,WC} <: AbstractProblem
    problem::P
    size::S
    cntr_size::WC
end
function WindowedProblem(op::Tuple; 
    cntr_size::Integer=1; 
    θ=nothing,
    problem=Problem(op; θ=nothing),
)
    WindowedProblem(problem, size, cntr_size)
end

function compute(p::WindowedProblem, rast::RasterStack)
    map(_window_ranges(p, size(rast))) do wr
        g = Grid(p, rast[wr...])
        compute(p, g)
    end
end

function _window_ranges(wp::WindowedProblem, tilesize::NTuple{2,Int})
    ci = CartesianIndices(tilesize)
    # Create an array of ranges for retreiving each window
    Iterators.map(ci) do window_corner
        map(Tuple(window_corner), wp.size) do i, s
            i:min(s, i + r)
        end
    end
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
    TiledProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over tiles of windowed grids.

"""
@kwdef struct TiledProblem{P,R,M}
    problem::P
    ranges::R
    layout::M
end
function TiledProblem(p::AbstractProblem;
    target::Raster,
    radius::Real,
    overlap::Real,
)
    res = resolution(target)
    # Convert distances to pixels
    r = radius / res
    o = overlap / res
    s = r - o # Step between each tile corner
    # Get the corners of each tile
    ci = CartesianIndices(target)[begin:s:end, begin:s:end]
    # Create an array of ranges for retreiving each tile
    tile_ranges = map(ci) do tile_corner
        map(tile_corner, size(target)) do i, sz
            i:min(sz, i + r)
        end
    end
    # Create a mask to skip tiles that have no target cells
    mask = map(ci) do tile_starts
        # Retrive a tile
        tile = view(target, tile_ranges...)
        # If there are non-NaN cells above zero, keep the tile
        any(x -> !isnan(x) && x > zero(x), tile)
    end

    return TiledProblem(w, ranges, mask)
end

function compute(p::TiledProblem, rast::RasterStack)
    map(p.ranges, p.mask) do rs, m
        m || return nothing
        tile = rast[rs...]
        mask!(tile; with=tile)
        g = Grid(tile)
        outputs = compute(p, g)
        write(outputs, p.storage)
        nothing
    end
end
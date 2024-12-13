abstract type AbstractOperation end 
abstract type RSPOperation <: AbstractOperation end

abstract type AbstractProblem end

"""
    compute(o, grid::Union{Grid,GridRSP})

Compute operation `o` on a grid.
"""
function compute end

"""
    assess(o::Operation, g::Grid)

Assess the memory and compute requirements of operation
`o` on grid `g`. This can be used to indicate memory
and time reequiremtents on a cluster
"""
function assess end


"""
    SolverMode

Abstract supertype for ConScape solver modes.
"""
abstract type SolverMode end
"""
   MaterializedSolve()

Compute all operations on a fully materialised Z matrix.
"""
struct MatrixSolve <: SolverMode end
"""
   ColumnSolve()

Compute all operations column by column,
after precomputing lu decompositions.
"""
struct VectorSolve <: SolverMode end


"""
    Problem(ops...; mode, θ)

Combine multiple compute operations into a single object, 
to be run in the same job.
"""
@kwdef struct Problem{M<:SolverMode,O,T} <: AbstractProblem
    mode::M=MaterializedSolve()
    ops::O
    θ::T=nothing
end
Problem(args...; kw...) = Problem(args; kw...)
Problem(p::AbstractProblem; θ=nothing, mode=p.mode) = Problem(o.ops; mode, θ)

compute(p::Problem, g::Grid) = compute(p.mode, p, g::Grid)
# Use an iterative solver so the grid is not materialised
function compute(m::VectorSolve, p::Problem, g::Grid)
    P = _Pref(g.affinities)
    W = _W(P, θ, g.costmatrix)
    # Sparse lhs
    a = I - W
    # Dense rhs column
    b = zeros(eltype(a), size(g.affinities, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(a, b)
    linsolve = init(linprob)
    map(columns) do 
        fill!(b, zero(eltype(b)))
        # fill b with column val
        linsolve = LinearSolve.set_b(linsolve, b)
        sol = solve(linsolve)
        res = sol2.u
        compute(o, res)
    end
end
# Materialise the whole rhs matrix
function compute(m::MatrixSolve, o::Problem, g::Grid) 
    # We could just move all the GridRSP code here?
    grsp = GridRSP(g; allocs=o.allocs)
    map(o.ops) do op
        compute(m, op, grsp)
    end
end


"""
    WindowedProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.

"""
@kwdef struct WindowedProblem{O,SS,WS,WC} <: AbstractProblem
    op::O
    sourcesize::SS
    size::WS
    centers::WC
end
function WindowedProblem(op::O, radius::Tuple, windowcenters::AbstractMatrix; 
    θ::T=nothing
) where O<:Tuple 
    WindowedProblem{O}(Problem(op; θ), windowsize, windowcenters)
end

function compute(p::WindowedProblem, rasterstack)
    map(_window_grids(p, rasterstack)) do g
        compute(p, g)
    end
end


"""
    TiledProblem(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over tiles of windowed grids.

"""
@kwdef struct TiledProblem{O,SS,WS,WC}
    op::O
    ranges::R
    layout::M
end
function TiledProblem(w::WindowedProblem;
    target::Raster,
    radius::Real,
    overlap::Real,
) where O<:Tuple 
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

function compute(p::TiledProblem, rast)
    map(p.ranges, p.mask) do rs, m
        m || return nothing
        tile = rast[rs...]
        g = Grid(tile)
        outputs = compute(p, w)
        write(outputs, p.storage)
        nothing
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

# @kwdef struct ComputeAssesment{P,M,T}
#     problem::P
#     mem_stats::M
#     totalmem::T
# end

# """
#     allocate(co::ComputeAssesment)

# Allocate memory required to run `compute` for the assessed ops.

# The returned object can be passed as the `allocs` keyword to `compute`.
# """
# function allocate(co::ComputeAssesment)
#     zmax = co.zmax
#     # But actually do this with GenericMemory using Julia v1.11
#     Z = Matrix{Float64}(undef, co.zmax) 
#     S = sparse(1:zmax[1], 1:zmax[2], 1.0, zmax...)
#     L = lu(S)
#     # Just return a NamedTuple for now
#     return (; Z, S, L)
# end

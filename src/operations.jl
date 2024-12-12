abstract type AbstractOperation end 
abstract type RSPOperation <: AbstractOperation end

@kwdef struct ComputeAssesment{O,M,T}
    op::O
    mem_stats::M
    totalmem::T
end

"""
    allocate(co::ComputeAssesment)

Allocate memory required to run `compute` for the assessed ops.

The returned object can be passed as the `allocs` keyword to `compute`.
"""
function allocate(co::ComputeAssesment)
    zmax = co.zmax
    # But actually do this with GenericMemory using Julia v1.11
    Z = Matrix{Float64}(undef, co.zmax) 
    S = sparse(1:zmax[1], 1:zmax[2], 1.0, zmax...)
    L = lu(S)
    # Just return a NamedTuple for now
    return (; Z, S, L)
end

"""
    compute(o::Operation, grsp::GridRSP)

Compute operation `o` on precomputed grid `grsp`.
"""
function compute end

"""
    assess(o::Operation, g::Grid; grid_assessment)

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
struct MaterializedSolve <: SolverMode end
"""
   ColumnSolve()

Compute all operations column by column,
after precomputing lu decompositions.
"""
struct ColumnSolve <: SolverMode end
"""
   IterativeSolve()

Compute all operations iteratively over columns.
"""
struct IterativeSolve <: SolverMode end

abstract type AbstractProblem end

"""
    Problem(ops...; θ)

Combine multiple compute operations into a single object, 
to be run in the same job.
"""
@kwdef struct Problem{M<:SolverMode,O<:,T} <: AbstractProblem
    mode::M
    op::O
    θ::T=nothing
end
Operations(op::O; kw...) where O<:Union{Tuple,NamedTuple} = Operations{O}(op; kw...)
Operations(args...; kw...) = Operations(args; kw...)
Operations(::Operations; θ=nothing) = Operations(o.op; θ)

compute(o::Operation, g::Grid) = comput(mode(o), o, g::Grid)
function compute(m::IterativeSolve, o::Operation, g::Grid)
    compute(o; θ=o.θ)
end
function compute(m::MaterializedSolve, o::Operations, g::Grid) 
    grsp = GridRSP(g; allocs=o.allocs)
    map(o.op) do op
        compute(m, op, grsp)
    end
end
function compute(m::ColumnSolve, o::Operations, g::Grid)
    # Compute columns in parallel
    # Apply operations to each target column as they are generated
    # Combine target to complet matrices to output 
end

"""
    WindowedOperations(op; size, centers, θ)

Combine multiple compute operations into a single object, 
to be run over the same windowed grids.


"""
@kwdef struct WindowedProblem{O,SS,WS,WC} <: AbstractProblem
    op::O
    sourcesize::SS
    size::WS
    centers::WC
end
function WindowedProblem(op::O, sourcesize::Tuple, windowsize::Tuple, windowcenters::AbstractMatrix; 
    θ::T=nothing
) where O<:Tuple 
    WindowedProblem{O}(Problem(op; θ), sourcesize, windowsize, windowcenters)
end

function compute(o::WindowedProblem, source::AbstractMatrix, target::AbstractMatrix)
    compute(o, GridRSP(g; allocs=o.allocs); θ=o.θ)
end

function assess(op::WindowedProblem, g::Grid) 
    window_assessments = map(_windows(op, g)) do w
        ca = assess(op.op, w)
    end
    maximums = reduce(window_assessments) do acc, a
        (; totalmem=max(acc.totalmem, a.totalmem),
           zmax=max(acc.zmax, a.zmax),
           lumax=max(acc.lumax, a.lumax),
        )
    end
    sums = reduce(window_assessments) do acc, a
        (; flops=acc.flops + a.flops)
    end
    ComputeAssesment(; op=op.op, maximums..., sums...)
end
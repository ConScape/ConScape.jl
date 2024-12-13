abstract type AbstractProblem end

"""
    compute(problem, grid::Union{Grid,GridRSP})

Compute problem `o` on a grid.
"""
function compute end

"""
    assess(p::AbstractProblem, g)

Assess the memory and compute requirements of problem
`p` on grid `g`. This can be used to indicate memory
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

This is inneficient for CPUS, but may be best for GPUs
using CuSSP.jl
"""
struct MatrixSolve <: SolverMode end
"""
   ColumnSolve()

Compute all operations column by column,
after precomputing lu decompositions.

TODO we can put solver details 
"""
struct VectorSolve <: SolverMode end

"""
    Problem(ops...; mode, θ)

Combine multiple compute operations into a single object, 
to be run in the same job.
"""
@kwdef struct Problem{M<:SolverMode,O,T} <: AbstractProblem
    mode::M=VectorSolve()
    ops::O
    θ::T=nothing
end
Problem(args...; kw...) = Problem(args; kw...)
Problem(p::AbstractProblem; θ=nothing, mode=p.mode) = Problem(o.ops; mode, θ)

function compute(p::Problem, rast::RasterStack)
    g = Grid(p, rast)
    compute(p, g)
end
compute(p::Problem, g::Grid) = compute(p.mode, p, g::Grid)

# Use an iterative solver so the grid is not materialised
function compute(m::VectorSolve, p::AbstractProblem, g::Grid)
    # Compute Z column by column
    _, targetnodes = _targetidx_and_nodes(g)
    P = _Pref(g.affinities)
    W = _W(P, θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    b = sparse(targetnodes,
        1:length(targetnodes),
        1.0,
        size(g.costmatrix, 1),
        length(targetnodes),
    )
    # Dense rhs column
    b = zeros(eltype(a), size(g.affinities, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob)
    map(1:size(a, 2)) do i 
        b .= view(A, i)
        # Update solver with new b values
        linsolve = LinearSolve.set_b(linsolve, b)
        sol = solve(linsolve)
        res = sol2.u
        # compute(p, res)
    end
end
# Materialise the whole rhs matrix
function compute(m::MatrixSolve, o::AbstractProblem, g::Grid) 
    # Legacy code... but maybe materialising is faster for CUDSS?
    Pref = _Pref(g.affinities)
    W    = _W(Pref, θ, g.costmatrix)
    targetidx, targetnodes = _targetidx_and_nodes(g)
    Z    = (I - W) \ Matrix(sparse(targetnodes,
                                 1:length(targetnodes),
                                 1.0,
                                 size(g.costmatrix, 1),
                                 length(targetnodes)))
    # Check that values in Z are not too small:
    if minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end

    map(o.ops) do op
        compute(m, op, grsp)
    end
end

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

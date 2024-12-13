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
using CuSSP.jl ?
"""
struct MatrixSolve <: SolverMode end
"""
   ColumnSolve()

Compute all operations column by column,
after precomputing lu decompositions.

TODO we can put LinearSolve.jl solver objects in the struct
"""
struct VectorSolve <: SolverMode end

"""
    Problem(ops...; mode, θ)

Combine multiple compute operations into a single object, 
to be run in the same job.
"""
@kwdef struct Problem{O,M<:SolverMode,T} <: AbstractProblem
    ops::O
    mode::M=VectorSolve()
    θ::T=1.0
end
Problem(ops::AbstractOperation...; kw...) = Problem(ops; kw...)
Problem(ops::Tuple; kw...) = Problem(; ops, kw...)
Problem(ops::NamedTuple; kw...) = Problem(; ops, kw...)
Problem(p::AbstractProblem; mode=p.mode, θ=nothing) = Problem(o.ops, mode, θ)

compute(p::Problem, rast::RasterStack) = compute(p, Grid(p, rast))
compute(p::Problem, g::Grid) = compute(p.mode, p, g::Grid)

# Use an iterative solver so the grid is not materialised
function compute(m::VectorSolve, p::AbstractProblem, g::Grid)
    # Compute Z column by column
    _, targetnodes = _targetidx_and_nodes(g)
    Pref = _Pref(g.affinities)
    W = _W(Pref, p.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse diagonal rhs matrix
    B = sparse(targetnodes,
        1:length(targetnodes),
        1.0,
        size(g.costmatrix, 1),
        length(targetnodes),
    )
    # Dense rhs column
    b = zeros(eltype(A), size(B, 1))
    @show size(A) size(B) size(b)

    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    Z = Matrix{eltype(A)}(undef, size(B))
    res = map(1:size(B, 2)) do i 
        b .= view(B, :, i)
        # Update solver with new b values
        linsolve = LinearSolve.set_b(linsolve, b)
        sol = solve(linsolve)
        # compute(op, g, sol.u, i) # aim for something like this ?
        Z[:, i] .= sol.u
        sol.u
    end
    # return _combine(res, g) # return results as Rasters

    # TODO remove all use of GridRSP
    grsp = GridRSP(g, p.θ, Pref, W, Z)
    return map(p.ops) do op
        compute(op, grsp)
    end
end
# Materialise the whole rhs matrix
function compute(m::MatrixSolve, p::AbstractProblem, g::Grid) 
    # Legacy code... but maybe materialising is faster for CUDSS?
    Pref = _Pref(g.affinities)
    W    = _W(Pref, p.θ, g.costmatrix)
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
    grsp = GridRSP(g, p.θ, Pref, W, Z)
    return map(p.ops) do op
        compute(op, grsp)
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

connectivity_measure(p::AbstractProblem) = p.connectivity_measure
connectivity_function(p::AbstractProblem) =
    connectivity_function(connectivity_measure(p))
distance_transformation(p::AbstractProblem) = p.distance_transformation
    

"""
    solve(problem, grid::Union{Grid,GridRSP})

Solve problem `o` for a grid.
"""
function solve end

"""
    assess(p::AbstractProblem, g)

Assess the memory and solve requirements of problem
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

solve all operations on a fully materialised Z matrix.

This is inneficient for CPUS, but may be best for GPUs
using CuSSP.jl ?
"""
struct MatrixSolve <: SolverMode end
"""
   VectorSolve()

Solve all operations column by column,
after precomputing lu decompositions.

TODO we can put LinearSolve.jl solver objects in the struct
"""
struct VectorSolve{T} <: SolverMode 
    keywords::T
end

"""
    Problem(graph_measures...; solver, θ)

Combine multiple solve operations into a single object, 
to be run in the same job.
"""
@kwdef struct Problem{GM,CM<:ConnectivityMeasure,DT,SM<:SolverMode} <: AbstractProblem
    graph_measures::GM
    connectivity_measure::CM=LeastCostDistance()
    distance_transformation::DT=nothing
    solver::SM=VectorSolve((;))
end
Problem(gms::GraphMeasure...; kw...) = Problem(gms; kw...)
Problem(graph_measures::Union{Tuple,NamedTuple}; kw...) = Problem(; graph_measures, kw...)
Problem(p::AbstractProblem; solver=p.solver, θ=nothing) = Problem(o.graph_measures, solver, θ)

solve(p::Problem, rast::RasterStack) = solve(p, Grid(p, rast))
solve(p::Problem, g::Grid) = solve(p.solver, connectivity_measure(p), p, g::Grid)


# Use an iterative solver so the grid is not materialised
function solve(m::VectorSolve, cm::FundamentalMeasure, p::AbstractProblem, g::Grid)
    # solve Z column by column
    Pref = _Pref(g.affinities)
    W = _W(Pref, cm.θ, g.costmatrix)
    # Sparse lhs
    A = I - W
    # Sparse diagonal rhs matrix
    B = sparse(g.targetnodes,
        1:length(g.targetnodes),
        1.0,
        size(g.costmatrix, 1),
        length(g.targetnodes),
    )
    b_init = zeros(eltype(A), size(B, 1))
    # Dense rhs column

    # Define and initialise the linear problem
    linprob = LinearProblem(A, b_init)
    linsolve = init(linprob)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    Z = Matrix{eltype(A)}(undef, size(B))
    nbuffers = Threads.nthreads()
    # Create a channel to store problem b vectors for threads
    # I'm not sure if this is the most efficient way
    # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    ch = Channel{Tuple{typeof(linsolve),Vector{Float64}}}(nbuffers)
    for i in 1:nbuffers
        # TODO fix this in LinearSolve.jl with batching
        # We should not need to `deepcopy` the whole problem we 
        # just need to replicate the specific workspace arrays 
        # that will cause race conditions.
        # But currently there is no parallel mode for LinearSolve.jl
        # See https://github.com/SciML/LinearSolve.jl/issues/552
        put!(ch, (deepcopy(linsolve), Vector{eltype(A)}(undef, size(B, 1))))
    end
    Threads.@threads for i in 1:size(B, 2)
        # Get column memory from the channel
        linsolve1, b = take!(ch)
        # Update it
        b .= view(B, :, i)
        # Update solver with new b values
        linsolve2 = LinearSolve.set_b(linsolve1, b)
        sol = LinearSolve.solve(linsolve2)
        # Aim for something like this ?
        # res = map(connectivity_measures(p)) do cm
        #     compute(cm, g, sol.u, i)
        # end

        # For now just use Z
        Z[:, i] .= sol.u
        put!(ch, (linsolve1, b))
    end
    # return _combine(res, g) # return results as Rasters

    # TODO remove all use of GridRSP
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    return map(p.graph_measures) do gm
        compute(gm, p, grsp)
    end
end
# Materialise the whole rhs matrix
function solve(m::MatrixSolve, cm::FundamentalMeasure, p::AbstractProblem, g::Grid) 
    # Legacy code... but maybe materialising is faster for CUDSS?
    Pref = _Pref(g.affinities)
    W    = _W(Pref, cm.θ, g.costmatrix)
    Z    = (I - W) \ Matrix(sparse(g.targetnodes,
                                 1:length(g.targetnodes),
                                 1.0,
                                 size(g.costmatrix, 1),
                                 length(g.targetnodes)))
    # Check that values in Z are not too small:
    if minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
    grsp = GridRSP(g, cm.θ, Pref, W, Z)
    return map(p.graph_measures) do gm
        compute(gm, grsp)
    end
end
function solve(::SolverMode, cm::ConnectivityMeasure, p::AbstractProblem, g::Grid) 
    # GridRSP is not used here
    return map(p.graph_measures) do gm
        compute(gm, p, g)
    end
end

# @kwdef struct ComputeAssesment{P,M,T}
#     problem::P
#     mem_stats::M
#     totalmem::T
# end

# """
#     allocate(co::ComputeAssesment)

# Allocate memory required to run `solve` for the assessed ops.

# The returned object can be passed as the `allocs` keyword to `solve`.
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

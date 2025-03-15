# TODO: move all the preallocation in solve.jl here

"""
    GridPrecalculations

Abstract type for precalculated variables for use in graph measures.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables once for all targets of multiple
graph measure.
"""
abstract type GridPrecalculations{CM} end

grid(gp::GridPrecalculations) = gp.grid
problem(gp::GridPrecalculations) = problem(gp.problem)
solver(gp::GridPrecalculations) = solver(problem(gp))

Base.size(gp::GridPrecalculations) = size(grid(gp))
DimensionalData.dims(gp::GridPrecalculations) = dims(grid(gp))

"""
    TargetPrecalculations

Abstract type for precalculated variables at the level of single targets.
"""
abstract type TargetPrecalculations{CM} end

grid_precalculations(gp::TargetPrecalculations) = gp.grid_precalculations
grid(gp::TargetPrecalculations) = grid(grid_precalculations(gp))
problem(gp::TargetPrecalculations) = problem(grid_precalculations(gp))
solver(gp::TargetPrecalculations) = solver(grid_precalculations(gp))

Base.size(gp::TargetPrecalculations) = size(grid_precalculations(gp))
DimensionalData.dims(gp::TargetPrecalculations) = dims(grid_precalculations(gp))

"""
    RandomisedShortestPathPrecalculations(g::Grid; θ=nothing)

Stores precalculated variables for use in `RandomisedShortestPath`-based measures.

(formerly GridRSP)
"""
struct RandomisedShortestPathGridPrecalculations{CM,P<:Problem{CM},S,F,Sadj,Fadj} <: GridPrecalucations
    grid::Grid
    problem::P
    probability::S
    W::S # TODO: better field names
    IW::S
    IW_factorization::F
    IW_adj::Sadj
    IW_adj_factorization::Fadj
end

function init(
    ::RandomisedShortestPath, problem::Problem, rast::RasterStack
)
    g = Grid(problem, rast)
    cm = connectivity_measure(m)
    probability = _probabilities(g.affinities)
    W = _W(probability, θ, g.costmatrix)
    IW = I - W
    IW_factorization = init(solver, IW)
    IW_adj, IW_adj_factorization = if hastrait(needs_adjoint_init, measures(problem))
        # Just take the adjoint of the factorization of A
        # where possible to save calculations and memory
        if hasproperty(A_init, :F)
            Aadj = A'
            # Use adjoint factorization of A rather than recalculating for A'
            Aadj_init = merge(A_init, (; F=A_init.F'))
            Aadj, Aadj_init
        else
            # LinearSolve.jl cant handle the adjoint 
            # so we duplicate work and allocations
            Aadj = sparse(A')
            Aadj_init = init(solver, Aadj)
            Aadj, Aadj_init
        end
    else
        nothing, nothing
    end
    CW = grid.costmatrix .* W

    return RandomisedShortestPathGridPrecalculations(
        cm, g, problem, probability, W, IW, IW_factorization, IW_adj, IW_adj_factorization, CW
    )
end

struct RandomisedShortestPathTargetPrecalculations{CM,GP<:RandomisedShortestPathGridPrecalculations{CM},D<:AbstractMatrix}
    grid_precalculations::GP
    fundamental_matrix::D
    proximities::D
    landscape_matrix::D
    target::Target
end

function init(
    gp::RandomisedShortestPathGridPrecalculations, 
    target::Target,
)
    workspace1, workspace2 = workspaces(gp)
    B_sparse = _sparse_rhs(target.node:target.node, size(g.costmatrix, 1))
    B = workspace1 .= B_sparse
    B_copy = workspace2 .= B
    fundamental_matrix = ldiv!(solver(gp), gp.IW, B; B_copy)
    proximities = compute(connectivity_measure(gp), gp)
    RandomisedShortestPathTargetPrecalculations(gp, fundamental_matrix, proximities, target)
end


"""
    LeastCostPrecalculations(g::Grid)

Stores precalculated variables for use in `LeastCost`-based measures.
"""
struct LeastCostGridPrecalculations{CM,P<:Problem{CM},S} <: GridPrecalucations
    grid::Grid
    problem::P
    probability::S
    cost_weighted_digraph::SimpleWeightedDiGraph{Float64}
end

function init(
    ::LeastCost, problem::Problem, rast::RasterStack
)
    g = Grid(problem, rast)
    probability = _Pref(g.affinities)
    cost_weighted_digraph = simpleweighteddigraph(g.costmatrix)
    LeastCostGridPrecalculations(g, cm, problem, probability, cost_weighted_digraph)
end

struct LeastCostTargetPrecalculations{CM,GP<:LeastCostGridPrecalculations{CM}} <: TargetPrecalucations
    grid_precalculations::GP
    target::Target
end

function init(gp::LeastCostGridPrecalculations, target::Target)
    LeastCostTargetPrecalculations(gp, target)
end

"""
    RandomWalkPrecalculations(g::Grid)

Stores precalculated variables for use in `RandomWalk`-based measures.
"""
struct RandomWalkGridPrecalculations{CM,P<:Problem{CM},S} <: GridPrecalucations
    grid::Grid
    problem::P
    probability::S
end

function init(
    m::RandomWalk, problem::Problem, rast::RasterStack
)
    g = Grid(problem, rast)
    probability = _probabilities(g.affinities)
    workspaces = _workspaces(problem, size(g.costmatrix, 1))

    # fundamental = ldiv!(solver(gp), gp.IW, B; B_copy)
    RandomWalkPrecalculations(g, problem, probability)
end

struct RandomWalkTargetPrecalculations{CM,GP<:RandomWalkGridPrecalculations{CM},D,SD}
    grid_precalculations::GP
    fundamental::D
    hitting_time::D
    stationary_distribution::SD
    target::Target
end
function init(grid_precalculations::RandomWalkTargetPrecalculations, target::Target)
     # TODO make this single target, not square
    fundamental_matrix = inv(Matrix(I - P) .+ p')
    hitting_time = (diag(Z)' .- Z) ./ p'
    RandomWalkTargetPrecalculations(
        grid_precalculations, 
        fundamental_matrix, 
        hitting_time, 
        stationary_distribution, 
        target,
    )
end

# Computes the stationary distribution of a random walk following the transition probability matrix
function stationary_distribution(P::SparseMatrixCSC, solver::AbstractSolver)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI[1, :] .= 1
    v = zeros(n)
    v[1] = 1
    return ldiv!(solver, PI, v)
end

function _computeproximities(g::Grid, kw...
)
    proximities = connectivity_function(g; kw...)
    if connectivity_function <: DistanceFunction
        map!(distance_transformation, proximities, proximities)
    end
    maybe_set_diagonal!(proximities, diagvalue, g.targetnodes)
    return proximities
end

maybe_set_diagonal!(proximities, diagvalue::Nothing, targetnodes::AbstractVector) = nothing
function maybe_set_diagonal!(proximities, diagvalue, targetnodes::AbstractVector)
    for (j, i) in enumerate(targetnodes)
        proximities[i, j] = diagvalue
    end
end

# Generate the sparse diagonal rhs matrix
function _sparse_rhs(targetnodes, n)
    m = length(targetnodes)
    sparse(targetnodes, 1:m, 1.0, n, m)
end

function _probabilities(A::SparseMatrixCSC)
    # TODO drop the LinAlg here, broadcasting a division by 
    # a vector is faster, and easier to read for almost everyone
    source_sums = vec(sum(A, dims=2))
    source_scaling = inv.(source_sums)
    return Diagonal(source_scaling) * A
end

function _W(Pref::SparseMatrixCSC, θ::Real, C::SparseMatrixCSC)
    n = LinearAlgebra.checksquare(Pref)
    W = Pref .* exp.((-).(θ) .* C)
    replace!(W.nzval, NaN => 0.0)
    return W
end

_inv(Z) = _inv!(similar(Z), Z)
function _inv!(Zⁱ, Z)
    broadcast!(Zⁱ, Z) do x
        x = inv(x)
        isfinite(x) ? x : floatmax(eltype(Z))
    end
end

function _workspaces(problem::Problem, size)
    map(1:nworkspaces(problem(gp))) do i
        Matrix{Float64}(undef, size(B_sparse))
    end
end
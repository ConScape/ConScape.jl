# TODO: move all the preallocation in solve.jl here

"""
    GridPrecalculations

Abstract type for precalculated variables for use in graph measures.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables once for all targets of multiple
graph measure.
"""
abstract type GridPrecalculations end

grid(gp::GridPrecalculations) = gp.grid
Base.size(gp::GridPrecalculations) = size(grid(gp))
DimensionalData.dims(gp::GridPrecalculations) = dims(grid(gp))

"""
    TargetPrecalculations

Abstract type for precalculated variables at the level of single targets.
"""
abstract type TargetPrecalculations end

"""
    precalculate(::MovementMode, g::Grid)

Returns a `GridPrecalculation` object for the specific `MovementMode`.

This contains all sparse and dense matrices needed for computing 
graph measures for the targets in the `Grid` `g`.
"""
function precalculate end

"""
    RandomisedShortestPathPrecalculations(g::Grid; θ=nothing)

Stores precalculated variables for use in `RandomisedShortestPath`-based measures.

(formerly GridRSP)
"""
struct RandomisedShortestPathGridPrecalculations{CM,S<:AbstractSolver,F} <: GridPrecalucations
    g::Grid
    cm::CM
    solver::S
    θ::Float64
    probability::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int} # TODO what is a longer name for W
    IW::SparseMatrixCSC{Float64,Int}
    IW_factorization::F
    # TODO the rest here
end
function RandomisedShortestPathGridPrecalculations(
    m::RandomisedShortestPath, problem::Problem, rast::RasterStack
)
    g = Grid(problem, rast)
    cm = connectivity_measure(m)
    probability = _probabilities(g.affinities)
    W = _W(probability, θ, g.costmatrix)
    IW = I - W
    IW_factorization = init(solver, IW)

    # Check that values in Z are not too small:
    verbose && if minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end

    return RandomisedShortestPathGridPrecalculations(cm, g, θ, probability, W, IW, IW_factorization, fundamental, solver)
end

precalculate(m::RandomisedShortestPath, g) = RandomisedShortestPathGridPrecalculations(g, m)
precalculate(gp::RandomisedShortestPathGridPrecalculations, targets) = 
    RandomisedShortestPathTargetPrecalculations(gp, targets)

struct RandomisedShortestPathTargetPrecalculations{CM}
    gp::RandomisedShortestPathGridPrecalculations{CM}
    fundamental_matrix::Matrix{Float64}
    proximities::Matrix{Float64}
    landscape_matrix::Matrix{Float64}
    target::Target
end
function RandomisedShortestPathTargetPrecalculations(
    gp::RandomisedShortestPathGridPrecalculations, 
    target::Target,
    nworkspaces=1
)
    B_sparse = _rsp_sparse_rhs(g.targetnodes, size(g.costmatrix, 1))
    workspaces = map(1:nworkspaces) do i
        Matrix{Float64}(undef, size(B_sparse))
    end
    B = Matrix(B_sparse)
    B_copy=copyto!(workspaces[1], B)
    fundamental = ldiv!(solver(gp), gp.IW, B; B_copy)
    proximities = compute(cm, gp)
    RandomisedShortestPathTargetPrecalculations(gp, fundamental, proximities, target)
end

# Update rhs
function reinit!(allocs::RandomisedShortestPathGridPrecalculations, g; θ=nothing)
end

# Generate the sparse diagonal rhs matrix
function _rsp_sparse_rhs(targetnodes, n)
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


"""
    LeastCostPrecalculations(g::Grid)

Stores precalculated variables for use in `LeastCost`-based measures.
"""
struct LeastCostPrecalculations <: GridPrecalucations
    g::Grid
    probability::SparseMatrixCSC{Float64,Int}
    cost_weighted_digraph::SimpleWeightedDiGraph{Float64}
    # TODO the rest here
end
function LeastCostPrecalculations(g)
    probability = _Pref(g.affinities)
    cost_weighted_digraph = simpleweighteddigraph(g.costmatrix)
    LeastCostPrecalculations(g, probability, cost_weighted_digraph)
end

precalculate(::LeastCost, g) = LeastCostPrecalculations(g)

function reinit!(allocs::LeastCostPrecalculations, g)
end

"""
    RandomWalkPrecalculations(g::Grid)

Stores precalculated variables for use in `RandomWalk`-based measures.
"""
struct RandomWalkGridPrecalculations{CM,S} <: GridPrecalucations
    g::Grid
    connectivity_measure::CM
    solver::S
    probability::SparseMatrixCSC{Float64,Int}
    fundamental::Matrix{Float64}
    hitting_time::Matrix{Float64}
    stationary_distribution::Vector{Float64}
    # TODO the rest here
end
function RandomWalkGridPrecalculations(g; solver=VectorSolver())
    probability = _probabilities(g.affinities)
    stationary_distribution = stationary_distribution(probability)
    fundamental = inv(Matrix(I-P) .+ p') # TODO make this not square
    hitting_time = (diag(Z)' .- Z) ./ p'
    RandomWalkPrecalculations(g, probability, fundamental, hitting_time, stationary_distribution, solver)
end

struct RandomWalkTargetPrecalculations{CM,S}
    gp::RandomWalkGridPrecalculations{CM,S}
    target::Target
end
function RandomWalkTargetPrecalculations(
    gp::RandomWalkGridPrecalculations, 
    target::Target,
)
    RandomWalkTargetPrecalculations(gp, target)
end

precalculate(m::RandomWalk, g) = RandomWalkGridPrecalculations(m, g)
precalculate(g::RandomWalkGridPrecalculation, target) = RandomWalkTargetPrecalculations(g, target)

function reinit!(allocs::RandomWalkPrecalculations, g)
    # TODO in-place version
end

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


function _computeproximities(grsp;
    connectivity_function=expected_cost,
    distance_transformation=nothing,
    diagvalue=nothing,
    kw...
)
    proximities = connectivity_function(g; kw...)

    # Check that distance_transformation function has been passed if no cost function is saved
    if connectivity_function <: DistanceFunction
        if distance_transformation === nothing
            if g.costfunction === nothing
                throw(ArgumentError("no distance_transformation function supplied and cost matrix in GridRSP isn't based on a cost function."))
            else
                distance_transformation = inv(g.costfunction)
            end
        end
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
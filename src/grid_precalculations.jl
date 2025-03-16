
struct GridInit{P,G,W,O} <: Precalculations
    problem::P
    grid::G
    subgrids::Vector{G}
    workspaces::Vector{W}
    outputs::O
end
function GridInit(problem::Problem, rast::RasterStack; 
    workspaces=nothing, verbose=false, coarse=nothing,
)
    grid = Grid(problem, rast)
    if !isnothing(coarse)
        grid = coarse_graining(grid, coarse)
    end
    subgrids = split_subgraphs(grid)
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    # Generate outputs for each graph measure
    outputs = map(measure -> allocate_output(measure, grid), graph_measures(problem))
    GridInit(problem, grid, subgrids, workspaces, outputs)
end

init(problem::Problem, rast::RasterStack; kw...) = GridInit(problem, rast; kw...)
init(gi::GridInit, subgrid_id::Int) =
    init(problem(gi), gi.subgrids[subgrid_id]; workspaces=workspaces(gi))
init(p::Problem, grid::Grid; kw...) = init(movement_mode(p), p, grid; kw...)

problem(gi::GridInit) = gi.problem
grid(gi::GridInit) = gi.grid
subgrids(gi::GridInit) = gi.subgrids
workspaces(gi::GridInit) = gi.workspaces
outputs(gi::GridInit) = gi.outputs

"""
    GridPrecalculations

Abstract type for precalculated variables for use in graph measures.

As we often iterate over single targets it is necessary to precalculate
and store expensive variables once for all targets of multiple
graph measure.
"""
abstract type GridPrecalculations <: Precalculations end

grid(gp::GridPrecalculations) = gp.grid
problem(gp::GridPrecalculations) = gp.problem
probability(gp::GridPrecalculations) = gp.probability
workspaces(gp::GridPrecalculations) = gp.workspaces

Base.size(gp::GridPrecalculations) = size(grid(gp))
DimensionalData.dims(gp::GridPrecalculations) = dims(grid(gp))

init(mm::MovementMode, problem::Problem, rast::RasterStack; kw...) = 
    init(mm, problem, GridInit(problem, rast); kw...)
# Allow updating a precalculated grid with a new grid.
# Basically this reuses workspace vectors
init!(gp::GridPrecalculations, rast::RasterStack; kw...) =
    init!(gp, GridInit(problem, rast); kw...)
init!(gp::GridPrecalculations, grid::Grid) = 
    init(movement_mode(gp), problem(gp), grid; workspaces=workspaces(gp))

# Solve defers to specific solver methods in solvers.jl
solve!(gp::GridPrecalculations, p::Problem; kw...) =
    solve!(solver(p), gp, p; kw...)

"""
    TargetPrecalculations

Abstract type for precalculated variables at the level of single targets.
"""
abstract type TargetPrecalculations end

grid_precalculations(tp::TargetPrecalculations) = tp.grid_precalculations
grid(tp::TargetPrecalculations) = grid(grid_precalculations(tp))
problem(tp::TargetPrecalculations) = problem(grid_precalculations(tp))
workspaces(tp::TargetPrecalculations) = workspaces(grid_precalculations(tp))
probability(tp::TargetPrecalculations) = probability(grid_precalculations(tp))
fundamental_matrix(tp::TargetPrecalculations) = tp.fundamental_matrix

Base.size(gp::TargetPrecalculations) = size(grid_precalculations(gp))
DimensionalData.dims(gp::TargetPrecalculations) = dims(grid_precalculations(gp))

"""
    RandomisedShortestPathPrecalculations(g::Grid; θ=nothing)

Stores precalculated variables for use in `RandomisedShortestPath`-based measures.

(formerly GridRSP)
"""
struct RandomisedShortestPathGridPrecalculations{P<:Problem,G,S,F,Sadj,Fadj,W} <: GridPrecalculations
    problem::P
    grid::G
    probability::S
    W::S # TODO: better field names
    IW::S
    CW::S
    IW_factorization::F
    IW_adj::Sadj
    IW_adj_factorization::Fadj
    workspaces::Vector{W}
end

function init(mm::RandomisedShortestPath, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    @show size(grid) sparse_size(grid)
    probability = _probabilities(affinitymatrix(grid))
    W = _W(probability, theta(mm), costmatrix(grid))
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    IW_adj, IW_adj_factorization = if solver(problem) isa VectorSolver
        IWadj = IW'
        # Use adjoint factorization of A rather than recalculating for A'
        IWadj_factorization = IW_factorization'
        IWadj, IWadj_factorization
    else # LinearSolver
        # LinearSolve.jl cant handle the adjoint 
        # so we duplicate work and allocations
        IWadj = sparse(A')
        IWadj_factorization = init(solver, IWadj)
        IWadj, IWadj_factorization
    end
    CW = costmatrix(grid) .* W
    workspaces = _allocate_workspaces!(workspaces, problem, grid)

    return RandomisedShortestPathGridPrecalculations(
        problem, grid, probability, W, IW, CW, IW_factorization, IW_adj, IW_adj_factorization, workspaces
    )
end

connectivity_measure(mm::RandomisedShortestPathGridPrecalculations) = connectivity_measure(problem(mm))
diagvalue(mm::RandomisedShortestPathGridPrecalculations) = diagvalue(problem(mm))
approx(mm::RandomisedShortestPathGridPrecalculations) = approx(problem(mm))
theta(mm::RandomisedShortestPathGridPrecalculations) = theta(problem(mm))

struct RandomisedShortestPathTargetPrecalculations{
    GP<:RandomisedShortestPathGridPrecalculations,D<:AbstractMatrix
} <: TargetPrecalculations
    grid_precalculations::GP
    fundamental_matrix::D
    proximities::D
    landscape_matrix::D
    target::TargetID
end

function init(
    gp::RandomisedShortestPathGridPrecalculations, 
    target::TargetID,
)
    println()
    @show length(source_ids(gp)) target
    workspace1, workspace2 = workspaces(gp)
    @show length(workspace1) size(grid(gp)) sparse_size(grid(gp))
    B = _rhs!(workspace1, length(source_ids(gp)), target)
    B_copy = _rhs!(workspace2, length(source_ids(gp)), target)
    fundamental_matrix = ldiv!(solver(gp), gp.IW, B; B_copy)
    proximities = compute(connectivity_measure(gp), gp)
    RandomisedShortestPathTargetPrecalculations(gp, fundamental_matrix, proximities, target)
end

connectivity_measure(mm::RandomisedShortestPathTargetPrecalculations) = connectivity_measure(problem(mm))
diagvalue(mm::RandomisedShortestPathTargetPrecalculations) = diagvalue(problem(mm))
approx(mm::RandomisedShortestPathTargetPrecalculations) = approx(problem(mm))
theta(mm::RandomisedShortestPathTargetPrecalculations) = theta(problem(mm))

"""
    LeastCostPrecalculations(g::Grid)

Stores precalculated variables for use in `LeastCost`-based measures.
"""
struct LeastCostGridPrecalculations{P<:Problem,G<:Grid,S,W} <: GridPrecalculations
    problem::P
    grid::G
    probability::S
    cost_weighted_digraph::SimpleWeightedDiGraph{Int,Float64}
    workspaces::Vector{W}
end

function init(::LeastCost, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    probability = _Pref(affinitymatrix(grid))
    cost_weighted_digraph = simpleweighteddigraph(costmatrix(grid))
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    LeastCostGridPrecalculations(problem, grid, probability, cost_weighted_digraph, workspaces)
end

struct LeastCostTargetPrecalculations{GP<:LeastCostGridPrecalculations} <: TargetPrecalculations
    grid_precalculations::GP
    target::TargetID
end

function init(gp::LeastCostGridPrecalculations, target::TargetID)
    LeastCostTargetPrecalculations(gp, target)
end

"""
    RandomWalkPrecalculations(g::Grid)

Stores precalculated variables for use in `RandomWalk`-based measures.
"""
struct RandomWalkGridPrecalculations{P<:Problem,G<:Grid,S,W} <: GridPrecalculations
    problem::P
    grid::G
    probability::S
    workspaces::Vector{W}
end

function init(::RandomWalk, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    probability = _probabilities(affinitymatrix(grid))
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    RandomWalkGridPrecalculations(problem, grid, probability, workspaces)
end

struct RandomWalkTargetPrecalculations{
    GP<:RandomWalkGridPrecalculations,D,SD
} <: TargetPrecalculations
    grid_precalculations::GP
    fundamental::D
    hitting_time::D
    stationary_distribution::SD
    target::TargetID
end
function init(grid_precalculations::RandomWalkTargetPrecalculations, target::TargetID)
    workspace1, workspace2 = workspaces(grid_precalculations)
     # TODO make this single target, not square
    fundamental_matrix = workspace1 .= inv(Matrix(I - P) .+ p')
    hitting_time = workspace2 .= (diag(Z)' .- Z) ./ p'
    RandomWalkTargetPrecalculations(
        grid_precalculations, 
        fundamental_matrix, 
        hitting_time, 
        stationary_distribution, 
        target,
    )
end

# Computes the stationary distribution of a random walk following the transition probability matrix
function stationary_distribution(P::SparseMatrixCSC, solver::Solver)
    # Input: the transition probability matrix P
    # Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P)
    PI = P' - I
    PI[1, :] .= 1
    v = zeros(n)
    v[1] = 1
    return ldiv!(solver, PI, v)
end

maybe_set_diagonal!(proximities, diagvalue::Nothing, targetnodes::AbstractVector) = nothing
function maybe_set_diagonal!(proximities, diagvalue, targetnodes::AbstractVector)
    for (j, i) in enumerate(targetnodes)
        proximities[i, j] = diagvalue
    end
end

# Fill a vector with zeros, and one for the target node
function _rhs!(workspace, n::Int, target::TargetID)
    fill!(workspace, 0.0)
    workspace[target.node] = 1.0
    return workspace
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

# This duplicats some logic from gridrsp
function _proximities!(
    expected_costs::AbstractMatrix,
    gp::GridPrecalculations
)
    proximities = workspace1 
    dt = distance_transformation(gp)
    if isnothing(dt)
        proximities .= inv(g.costfunction).(expected_costs)
    else
        proximities .= dt.(expected_costs)
    end
    maybe_set_diagonal!(proximities, diagvalue(gp), targetnodes(gp))
    return proximities
end

function _allocate_workspaces!(::Nothing, problem::Problem, grid::Grid)
    map(1:nworkspaces(problem)) do i
        Vector{Float64}(undef, nsources(grid))
    end
end
function _allocate_workspaces!(workspaces::Vector, problem::Problem, grid::Grid)
    map(workspaces) do ws
        resize!(ws, nsources(grid))
    end
end

# This only makes sense if arrays are sorted large to small
function _reshape(A::Array, size::Tuple{Vararg{Int}})
    len = prod(size)
    if Base.size(A) == size
        A
    else # if length(A) >= len
        # TODO make sure this doesn't allocate when the array is larger
        # We may need julia 1.11 to do this properly
        v = vec(A)
        resize!(v, len)
        reshape(v, size)
    end
end
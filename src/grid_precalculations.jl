struct GridInit{P,G,W,O} <: Precalculations
    problem::P
    grid::G
    subgrids::Vector{G}
    workspaces::Workspaces{W}
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

function solve(gi::GridInit; kw...)
    # Loop over unnconnected subgraphs (there may be only one)
    for subgrid_id in eachindex(subgrids(gi))
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        subgrid_precalc = init(gi, subgrid_id)
        # Then loop over targets
        for target_id in target_ids(subgrid_precalc)
            # Precalculate for this target and graph measures
            target_precalc = init(subgrid_precalc, target_id)
            # Compute everything for this target and graph measures
            foreach(graph_measures(gi), outputs(gi)) do measure, output
                # Compute a measure for this target
                v = compute(measure, target_precalc)
                # Write values to output object
                update_output!(output, measure, v, target_precalc)
            end
        end
    end
    out = _maybe_raster(outputs(gi), grid(gi))

    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end

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

"""
    TargetPrecalculations

Abstract type for precalculated variables at the level of single targets.
"""
abstract type TargetPrecalculations <: Precalculations end

grid_precalculations(tp::TargetPrecalculations) = getfield(tp, :grid_precalculations)
target(tp::TargetPrecalculations) = getfield(tp, :target)
storage(tp::TargetPrecalculations) = getfield(tp, :storage)

grid(tp::TargetPrecalculations) = grid(grid_precalculations(tp))
problem(tp::TargetPrecalculations) = problem(grid_precalculations(tp))
workspaces(tp::TargetPrecalculations) = workspaces(grid_precalculations(tp))

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
    workspaces::Workspaces{W}
end

function init(mm::RandomisedShortestPath, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    probability = _probabilities(affinitymatrix(grid))
    W = _W(probability, theta(mm), costmatrix(grid))
    IW = I - W
    IW_factorization = init(solver(problem), IW)
    if solver(problem) isa VectorSolver
        IW_adj = IW'
        # Use adjoint factorization of A rather than recalculating for A'
        IW_adj_factorization = IW_factorization'
    else # LinearSolver
        # LinearSolve.jl cant handle the adjoint 
        # so we duplicate work and allocations
        IW_adj = sparse(IW')
        IW_adj_factorization = init(solver(problem), IW_adj)
    end
    CW = costmatrix(grid) .* W
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    @assert length(workspaces) == nsources(grid) == sparse_size(grid)[1]

    return RandomisedShortestPathGridPrecalculations(
        problem, grid, probability, W, IW, CW, IW_factorization, IW_adj, IW_adj_factorization, workspaces
    )
end

struct RandomisedShortestPathTargetPrecalculations{
    GP<:RandomisedShortestPathGridPrecalculations,S<:Dict{Symbol}
} <: TargetPrecalculations
    grid_precalculations::GP
    storage::S
    target::TargetID
end

function init(gp::RandomisedShortestPathGridPrecalculations, target::TargetID)
    free!(workspaces(gp))
    return RandomisedShortestPathTargetPrecalculations(gp, storagedict(gp), target)
end

function Base.getproperty(tp::RandomisedShortestPathTargetPrecalculations, x::Symbol)
    if x === :workspace
        return take!(workspaces(tp))
    elseif x === :qᵗ
        return target_quality_vector(tp)[target(tp).id]
    elseif x === :qˢ
        return source_quality_vector(tp)
    elseif x === :A
        return affinitymatrix(tp)
    elseif x === :C
        return costmatrix(tp)
    elseif hasproperty(grid_precalculations(tp), x)
        return Base.getproperty(grid_precalculations(tp), x)
    end

    # Otherwise its a not part of precalculations.
    # it may be stored from, or we may need to generate it
    get!(storage(tp), x) do
        if x === :Z
            _fundamental_matrix(tp)
        elseif x === :Zⁱ
            _inv!(tp.workspace, tp.Z)
        elseif x === :QZⁱ
            (; qˢ, Zⁱ, qᵗ, workspace) = tp
            workspace .= qˢ .* Zⁱ .* qᵗ
        elseif x === :K
            _proximities(tp)
        elseif x === :M
            (; qˢ, K, qᵗ, workspace) = tp
            workspace .= qˢ .* K .* qᵗ
        elseif x === :MZⁱ 
            (; M, Zⁱ, workspace) = tp
            workspace .= M .* Zⁱ 
        elseif x === :Zrows
            (; IW_adj_factorization) = tp
            b, b_copy = workspaces(tp)
            _rhs!(b, nsources(tp), target(tp))
            _rhs!(b_copy, nsources(tp), target(tp))
            ldiv!(solver(tp), b, IW_adj_factorization, b_copy)
        elseif x === :expected_costs
            compute(ExpectedCost(), tp)
        elseif x === :free_energy_distances
            compute(FreeEnergyDistance(), tp)
        elseif x === :survival_probabilities
            compute(SurvivalProbability(), tp)
        elseif x === :power_mean_proximities
            compute(PowerMeanProximity(), tp)
        else
            error("Unknown property $x")
        end
    end
end

storagedict(p::Precalculations) = Dict{Symbol,typeof(first(workspaces(p)))}()

function _fundamental_matrix(tp::RandomisedShortestPathTargetPrecalculations)
    workspace1, workspace2 = workspaces(tp)
    b = _rhs!(workspace1, nsources(tp), target(tp))
    b_copy = _rhs!(workspace2, nsources(tp), target(tp))
    return ldiv!(solver(tp), b, tp.IW_factorization, b_copy)
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
    workspaces::Workspaces{W}
end

function init(::LeastCost, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    probability = _Pref(affinitymatrix(grid))
    cost_weighted_digraph = SimpleWeightedDigraph(costmatrix(grid))
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    LeastCostGridPrecalculations(problem, grid, probability, cost_weighted_digraph, workspaces)
end

struct LeastCostTargetPrecalculations{
    GP<:LeastCostGridPrecalculations,S<:Dict{Symbol}
} <: TargetPrecalculations
    grid_precalculations::GP
    storage::S
    target::TargetID
end

function init(gp::LeastCostGridPrecalculations, target::TargetID)
    free!(workspaces(gp))
    LeastCostTargetPrecalculations(gp, storagedict(tp), target)
end

"""
    RandomWalkPrecalculations(g::Grid)

Stores precalculated variables for use in `RandomWalk`-based measures.
"""
struct RandomWalkGridPrecalculations{P<:Problem,G<:Grid,Pref,SD,W} <: GridPrecalculations
    problem::P
    grid::G
    probability::Pref
    stationary_distrionution::SD
    workspaces::Workspaces{W}
end

function init(::RandomWalk, problem::Problem, grid::Grid;
    workspaces=nothing, verbose=false,
)
    stationary_distribution = stationary_distribution(gp.Pref, solver(gp))
    probability = _probabilities(affinitymatrix(grid))
    workspaces = _allocate_workspaces!(workspaces, problem, grid)
    RandomWalkGridPrecalculations(problem, grid, probability, stationary_distribution, workspaces)
end

struct RandomWalkTargetPrecalculations{
    GP<:RandomWalkGridPrecalculations,S<:Dict{Symbol}
} <: TargetPrecalculations
    grid_precalculations::GP
    storage::S
    target::TargetID
end
function init(gp::RandomWalkTargetPrecalculations, target::TargetID)
    # TODO: calculate in getproperty
    # fundamental_matrix = workspace1 .= inv(Matrix(I - gp.Pref) .+ p')
    # hitting_time = workspace2 .= (diag(Z)' .- Z) ./ p'
    RandomWalkTargetPrecalculations(
        grid_precalculations, storagedict(gp), storage, target,
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

maybe_set_diagonal!(proximities, diagvalue::Nothing, targetnodes) = nothing
function maybe_set_diagonal!(proximities, diagvalue::Number, targetnodes::AbstractVector)
    for (j, i) in enumerate(targetnodes)
        proximities[i, j] = diagvalue
    end
end
maybe_set_diagonal!(proximities, diagvalue::Number, targetnode::Int) = 
    proximities[targetnode] = diagvalue

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
function _proximities(tp::TargetPrecalculations)
    proximities = compute(connectivity_measure(tp), tp)
    dt = distance_transformation(tp)
    if !isnothing(dt)
        proximities .= dt.(proximities)
    end
    maybe_set_diagonal!(proximities, diagvalue(tp), target(tp).node)
    return proximities
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

_allocate_workspaces!(x, problem::Problem, grid::Grid) =
    _allocate_workspaces!(x, problem, nsources(grid))
_allocate_workspaces!(x::Nothing, problem::Problem, length::Int) =
    Workspaces(length, nworkspaces(problem) + 20)
_allocate_workspaces!(workspaces::Workspaces, ::Problem, length::Int) =
    (resize!(free!(workspaces), length); workspaces)
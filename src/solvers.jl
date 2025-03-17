"""
   MatrixSolver(; check)

Solve all operations on a fully materialised Z matrix.

This is fast but memory inneficient for CPUS, and isn't threaded.
But may be best for GPUs using CuSSP.jl ?
"""
@kwdef struct MatrixSolver <: Solver
    check::Bool = true
end

"""
   VectorSolver(; check, threaded)

Use julias default solver but broken into columns, with 
less memory use and the capacity for threading
"""
@kwdef struct VectorSolver <: Solver
    check::Bool = true
    threaded::Bool = false
end

"""
   LinearSolver(args...; threded, kw...)

Solve all operations column-by-column using LinearSolve.jl solvers.

The `threaded` keyword specifies if threads are used per target.
Other arguments and keywords are passed to `LinearSolve.solve` after the
problem object, like:

````julia
`LinearSolve.solve(linearproblem, args...; kw...)`
````

# Example

This example uses LinearSolve.jl wth `KrylovJL_GMRES` and a preconditioner.

TODO: an example that is realistic

````julia
using LinearSolve
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor()),
problem = ConScape.Problem(; 
    solver = LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I)))
    graph_measures = (;
        func=ConScape.ConnectedHabitat(),
        qbetw=ConScape.BetweennessQweighted(),
    ),
    connectivity_measure = ConScape.ExpectedCost(θ=1.0),
)
````
"""
struct LinearSolver{A,K} <: Solver
    args::A
    keywords::K
    threaded::Bool
end
LinearSolver(args...; threaded=false, kw...) = LinearSolver(args, kw, threaded)

# _init_dense! may be called multiple times from `init!`, for each thread
function _init_dense!(
    ws::NamedTuple,
    solver::Solver,
    cm::FundamentalMeasure,
    p::AbstractProblem,
    grid::Grid;
    verbose=false,
    reuse_output=false,
)
    verbose && println("Retreiving measures...")
    subgrids = split_subgraphs(grid)
    g = first(subgrids)
    gms = graph_measures(p)
    cf = connectivity_function(p)
    verbose && println("Defining sparse arrays...")
    # B_dense becomes Z
    verbose && println("Allocating workspaces...")
    sze = _workspace_size(solver, g)
    Z = if hastrait(needs_inv, gms)
        if haskey(ws, :Z)
            _reshape(ws.Z, sze)
        else
            Matrix{eltype(g.affinities)}(undef, sze)
        end
    else
        nothing
    end
    Zⁱ = if hastrait(needs_inv, gms)
        haskey(ws, :Zⁱ) ? _reshape(ws.Zⁱ, sze) : similar(Z)
    else
        nothing
    end
    n_workspaces = count_workspaces(p)
    n_permuted_workspaces = count_permuted_workspaces(p)
    workspaces = if haskey(ws, :workspaces)
        [_reshape(w, size(Z)) for w in ws.workspaces]
    else
        [similar(Z) for _ in 1:n_workspaces]
    end
    permuted_workspaces = if haskey(ws, :workspaces)
        [_reshape(pw, size(Z')) for pw in ws.permuted_workspaces]
    else
        [similar(Z') for _ in 1:n_permuted_workspaces]
    end
    # TODO these shouldn't have traits, it 
    # should be baked into the problem.
    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        haskey(ws, :expected_costs) ? _reshape(ws.expected_costs, size(Z)) : similar(Z)
    else
        nothing
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        haskey(ws, :free_energy_distances) ? _reshape(ws.free_energy_distances, size(Z)) : similar(Z)
    else
        nothing
    end
    proximities = if hastrait(needs_proximity, gms)
        haskey(ws, :proximities) ? _reshape(ws.proximities, size(Z)) : similar(Z)
    else
        nothing
    end
    # We don't re-use outputs
    outputs = if reuse_output && haskey(ws, :outputs)
        ws.outputs
    else
        if distance_transformation(cm) isa NamedTuple
            map(gms) do gm
                if needs_connectivity(gm)
                    map(distance_transformation(cm)) do dt
                        allocate_output(gm)
                    end
                else
                    allocate_output(gm)
                end
            end
        else
            map(gms) do gm
                allocate_output(gm)
            end
        end
    end

    verbose && println("Finished allocating...")

    return (; Z, Zⁱ, workspaces, permuted_workspaces, free_energy_distances, expected_costs, proximities, outputs, grid, subgrids)
end

# Do all the work shared accross outputs
function _solve_dense!(ws::NamedTuple, solver::Solver, cm, p::Problem;
    verbose=false
)
    (; grid, W, Pref, A, A_init, Aadj_init, Aadj) = ws
    gms = graph_measures(p)
    cf = connectivity_function(p)
    # Sparse rhs
    # TODO get rid of this allocation
    # For VectorSolver we can write values directly to B
    B_sparse = sparse_rhs(grid.targetnodes, size(grid.costmatrix, 1))
    Z = if hastrait(needs_Z, gms)
        # verbose && 
        B = _reshape(ws.Z, size(B_sparse))
        copyto!(B, B_sparse)
        verbose && println("Solving Z matrix...")
        # Check that values in Z are not too small:

        Z = ldiv!(solver, A_init, B; B_copy=copyto!(ws.workspaces[1], B))
        # verbose && _check_z(s, Z, W, g)
        Z
    end
    Zⁱ = if hastrait(needs_inv, gms)
        verbose && println("Inverting Z...")
        _inv!(_reshape(ws.Zⁱ, size(Z)), Z)
    end

    grsp = GridRSP(grid, cm.θ, Pref, W, Z)
    workspace = (; ws..., Pref, W, A, A_init, Aadj, Aadj_init, Z, Zⁱ)

    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        verbose && println("Calculating expected cost...")
        expected_costs = _reshape(ws.expected_costs, size(Z))
        ConScape.expected_cost(grsp; workspace..., expected_costs, solver)
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        verbose && println("Calculating free energy distance...")
        free_energy_distances = _reshape(ws.free_energy_distances, size(Z))
        ConScape.free_energy_distance(grsp; workspace..., free_energy_distances, solver)
    end

    return merge(ws, (; Pref, W, A, A_init, Aadj, Aadj_init, Z, Zⁱ, expected_costs, free_energy_distances))
end

isthreaded(s::Solver) = false
isthreaded(s::LinearSolver) = s.threaded
isthreaded(s::VectorSolver) = s.threaded

# Solver init
init(::Union{Nothing,MatrixSolver,VectorSolver}, A::AbstractMatrix) = (; F=lu(A))
function init(solver::VectorSolver, A::AbstractMatrix)
    F = lu(A)
    if isthreaded(solver)
        nbuffers = Threads.nthreads()
        channel = Channel{typeof(F)}(nbuffers)
        for _ in 1:nbuffers
            put!(channel, copy(F))
        end
        return channel
    else
        return F
    end
end
function init(solver::LinearSolver, A::AbstractMatrix)
    b = zeros(eltype(A), size(A, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, solver.args...; solver.keywords...)
    # TODO what is needed here?
    # Create a channel to store problem b vectors for threads
    # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    if isthreaded(solver)
        nbuffers = Threads.nthreads()
        channel = Channel{Tuple{typeof(linsolve),Vector{Float64}}}(nbuffers)
        for i in 1:nbuffers
            # TODO fix this in LinearSolve.jl with batching
            # We should not need to `deepcopy` the whole problem we 
            # just need to replicate the specific workspace arrays 
            # that will cause race conditions.
            # But currently there is no parallel mode for LinearSolve.jl
            # See https://github.com/SciML/LinearSolve.jl/issues/552
            put!(channel, (deepcopy(linsolve), Vector{eltype(A)}(undef, size(A, 2))))
        end
        return channel
    else
        return linsolve
    end
end

function LinearAlgebra.ldiv!(p::Precalculations, init, B)
    # Handle using a workspace instead of copying B
    B_copy = take!(workspaces(p)) .= B
    # Solve
    X = ldiv!(solver(p), B, init, B_copy)
    # Return the workspace to the pool
    put!(workspaces(p), B_copy)
    return X
end
function LinearAlgebra.ldiv!(s::LinearSolver, B, init, B_copy)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    if isthreaded(s)
        channel = init
        # Get column memory from the channel
        linsolve = take!(channel)
        # Update solver with new b values
        reinit!(linsolve; b=vec(B_copy), reuse_precs=true)
        sol = LinearSolve.solve!(vec(B), linsolve, s.args...; s.keywords...)
        vec(B) .= sol.u
        put!(channel, linsolve)
    else
        linsolve = init
        reinit!(linsolve; b=vec(B_copy), reuse_precs=true)
        sol = LinearSolve.solve(linsolve, s.args...; s.keywords...)
        vec(B) .= sol.u
    end
    return B
end
# LinearAlgebra.ldiv!(::Union{MatrixSolver,Nothing}, B, (; F), B_copy) =
    # ldiv!(B, F, B_copy)
function LinearAlgebra.ldiv!(s::VectorSolver, B, init, B_copy)
    if isthreaded(s)
        channel = init
        F = take!(channel)
        # Solve for the column
        ldiv!(B, F, B_copy)
        # Reuse the workspace 
        put!(channel, F)
    else
        F = init
        ldiv!(B, F, B_copy)
    end
    return B
end

function _check_z(s, Z, W, g)
    # Check that values in Z are not too small:
    if hasproperty(s, :check) && s.check && minimum(Z) * minimum(nonzeros(g.costmatrix .* W)) == 0
        @warn "Warning: Z-matrix contains too small values, which can lead to inaccurate results! Check that the graph is connected or try decreasing θ."
    end
end
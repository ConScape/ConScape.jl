module ConScapeLinearSolveExt

using ConScape
using LinearSolve
import CommonSolve
import LinearAlgebra

# Now LinearSolve.jl is imported, add a working LinearSolver constructor
ConScape.LinearSolver(args::A, kw::K, threaded::Bool) where {A<:Tuple,K} =
    LinearSolver{A,K}(args, kw, threaded)

function CommonSolve.init(solver::LinearSolver, A::AbstractMatrix)
    b = zeros(eltype(A), size(A, 2))
    # Define and initialise the linear problem
    linprob = LinearProblem(A, b)
    linsolve = init(linprob, solver.args...; solver.keywords...)
    # TODO what is needed here?
    # Create a channel to store problem b vectors for threads
    # see https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/tls/tls/
    if ConScape.isthreaded(solver)
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

function LinearAlgebra.ldiv!(s::LinearSolver, B, init, B_copy)
    # TODO: for now we define a Z matrix, but later modify ops 
    # to run column by column without materialising Z
    if ConScape.isthreaded(s)
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

end
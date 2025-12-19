"""
    EigMax <: Measure

    Eigmax(; kw...)

Compute the largest eigenvalue triple (left vector, value, and right vector)
of the quality-scaled proximities with respect to the distance/proximity measure
in the [`MovementMode`](@ref).

## Keywords

- `seed`: seed for the random seed for the left eigenvector solve. 
    Useful if exact floating point replication is needed accross runs.
- `tol`: tolerance, defaults to `1e-14`.
"""
@kwdef struct EigMax{S,T} <: Measure
    seed::S = nothing
    tol::T = 1e-14
end

computelevel(m::EigMax) = ConnectedGraphLevel()
returntrait(::EigMax) = ReturnCustom()
needs_eigmax(::EigMax, ::RSP) = true

# Allocate sqauare matrix of target * target size
function allocate_gridgraph_output(
    ::ReturnCustom,
    m::EigMax,
    ::GridGraph,
    connectedgraphs::Vector
)
    T = Tuple{Vector{Float64},Array{Float64,0},Vector{Float64}}
    o = Vector{T}(undef, length(connectedgraphs))
    return MeasureOutput(m, o)
end
function allocate_connectedgraph_output(
    ::ConnectedGraphLevel,
    ::ReturnCustom,
    m::EigMax,
    ::GridGraph,
    cg::ConnectedGraph,
    precalculation
)
    n = length(sourceids(cg))
    vʳ = fill(NaN, n)
    λ = fill(0.0)
    vˡ = zeros(n)
    o = (vʳ, λ, vˡ)
    return MeasureOutput(m, o)
end

function _compute_eigmax(em::EigMax, cgi::ConnectedGraphInit)
    (; C, W) = cgi
    m = ntargets(cgi)
    targetnodes = map(x -> x.node, targetids(cgi))
    nontargetnodes = setdiff(1:nsources(cgi), targetnodes)

    # We use views into one matrix vec_workspace for both M matrices
    M_all = mat_workspace(cgi)
    MλI_all = mat_workspace(cgi)
    MλI = view(MλI_all, 1:m, :)
    Mtarget = view(M_all, 1:m, :)
    Mnontarget = view(M_all, m+1:size(M_all, 1), :)

    for target in targetids(cgi)
        ti = TargetInit(cgi, target)
        idx = target.connectedgraphidx
        M = ti.M

        # Copy M to matrices
        Mtarget[:, idx] .= view(M, targetnodes)
        if size(Mnontarget, 1) > 0
            Mnontarget[:, idx] .= view(M, nontargetnodes)
        end
    end

    # size of the full problem
    n = nsources(connectedgraph(cgi))

    # use an Arnoldi based eigensolver to compute the largest
    # (absolute) eigenvalue and right vector (of submatrix)
    Fps = ArnoldiMethod.partialschur(Mtarget; nev=1, tol=em.tol)
    λ₀, vʳ₀ = ArnoldiMethod.partialeigen(Fps[1])

    # Assign to the output Ref for λ
    λ = λ₀[]

    # compute left eigenvector (of submatrix) by shift-invert
    copyto!(MλI, -λ * I) # This is just alloc free `MλI = Mtarget - λ * I`
    MλI .+= Mtarget
    F = lu!(MλI)
    # TODO: explain rand here in a comment
    rng = isnothing(em.seed) ? Random.MersenneTwister() : Random.MersenneTwister(em.seed)
    rhs = rand(rng, length(targetnodes))
    vˡ₀ = ldiv!(copy(rhs), F', copy(rhs))

    # Allocate output left and right eigenvectors
    vˡ = fill(NaN, nsources(cgi))
    vʳ = fill(NaN, nsources(cgi))

    # Assign to the full right vector
    vʳ[targetnodes] .= vʳ₀
    # Fill the gaps from sources that are not targets
    vʳ[nontargetnodes] .= Mnontarget * vʳ₀ / λ
    # Assign to the full left vector
    vˡ[targetnodes] .= vˡ₀
    vˡ[nontargetnodes] .= 0.0

    # Normalize by dividing by the absolute maximum value
    # (scaled eigenvalues are considered identical)
    rmul!(vˡ, inv(vˡ[findmax(abs, vˡ)[2]]))
    rmul!(vʳ, inv(vˡ[findmax(abs, vʳ)[2]]))

    # Remove numbers below zero (e.g. from numerical instability)
    # map!(x -> x < zero(x) ? zero(x) : x, vˡ)
    # map!(x -> x < zero(x) ? zero(x) : x, vʳ)

    # Put back the matrix vec_workspace
    put!(mat_workspaces(cgi), M_all)
    put!(mat_workspaces(cgi), MλI_all)

    return (vˡ, λ, vʳ)
end

# The actual compute_connectedgraph! call is trivial as its fully precomputed above
function compute_connectedgraph!(output::Tuple, ::EigMax, cgi::ConnectedGraphInit) 
    (vˡ, λ, vʳ) = cgi.eigmax

    output[1] .= vˡ
    output[2][] = λ
    output[3] .= vʳ

    return output
end

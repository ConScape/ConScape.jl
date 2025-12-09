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
num_matrix_workspaces(::EigMax) = 1


# Allocate sqauare matrix of target * target size
function allocate_gridgraph_output(
    ::ReturnCustom,
    m::EigMax,
    ::ConScapeProblem,
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
    ::ConScapeProblem,
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

function compute_connectedgraph!(::EigMax, cgi::ConnectedGraphInit)
    (; C, W) = cgi
    m = length(targetids(cgi))
    targetnodes = map(x -> x.node, targetids(cgi))
    nontargetnodes = setdiff(1:m, targetnodes)

    # We use views into one matrix workspace for both M matrices
    M = mworkspace(cgi)
    Mtarget = view(M, 1:m, 1:m)
    Mnontarget = view(M, m+1:m+length(nontargetnodes), 1:m)

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

    # Computing the left and right vectors is optional
    if em.right isa Right
        # assign to the full right vector
        vʳ[targetnodes] .= vʳ₀
        vʳ[nontargetnodes] .= Mnontarget * vʳ₀ ./ λ₀[1]
    end

    if em.left isa Left
        # compute left vector (of submatrix) by shift-invert
        F = lu(Mtarget - λ₀[1] * I)
        rng = isnothing(em.seed) ? MersenneTwister() : MersenneTwister(seed)
        # TODO: explain rand here in a comment
        vˡ₀ = ldiv!(F', rand(rng, length(targetnodes))) # This is a hack that ensures a square matrix
        rmul!(vˡ₀, inv(vˡ₀[1]))
        # assign to the full left vector
        vˡ[targetnodes] .= vˡ₀
    end

    # Assign to the output Ref for λ
    λ[] = λ₀[1]

    return (vˡ, λ[], vʳ)
end

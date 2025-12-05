"""
    EigenSide

Abstract supertype for sides of [`EigMax`](@ref).

These let us skip computation with `NoLeft` or `NoRight`.

Both are computed by default, e.g. `Left()` and `Right` are used.
"""
abstract type EigenSide end

struct Left <: EigenSide end
struct NoLeft <: EigenSide end
struct Right <: EigenSide end
struct NoRight <: EigenSide end

Base.Symbol(m::Measure) = nameof(typeof(m))
function Base.Symbol(m::SensitivityAnalysis) 
    Symbol(
        nameof(typeof(m)), :_, 
        nameof(typeof(wrt(m))), :_, 
        nameof(typeof(metric(m))), :_, 
        nameof(typeof(sensitivitytype(m)))
    )
end

"""
    EigMax <: Measure

    Eigmax(; kw...)

Compute the largest eigenvalue triple (left vector, value, and right vector) 
of the quality-scaled proximities with respect to the distance/proximity measure 
in the [`MovementMode`](@ref).

## Keywords

`seed`: seed for the random seed for the left eigenvector solve. Useful if exact floating point
    replication is needed accross runs.
`tol`: tolerance, defaults to `1e-14`.
`left`: whether to calculate left eigenvector. Defaults to `Left()`, but can be `NoLeft()`.
`left`: whether to calculate righ eigenvector. Defaults to `Right()`, but can be `NoRight()`.

The triple is always returned, but if `NoLeft` or `NoRight` are used the 
values contained in the left/right vecto will be zeros.


"""
@kwdef struct EigMax{L,R,S,T} <: Measure
    left::L=Left()
    right::R=Right()
    seed::S = nothing
    tol::T = 1e-14
end

returntrait(::EigMax) = ReturnCustom()

# Allocate sqauare matrix of target * target size
function allocate_output(
    l::GridGraphLevel,
    ::ReturnCustom,
    m::EigMax,
    ::ConScapeProblem,
    ::GridGraph,
    connectedgraphs::Vector
)
    T = Tuple{Vector{Float64},Array{Float64,0},Vector{Float64}}
    l => Vector{T}(undef, length(connectedgraphs))
end
function allocate_output(
    l::Union{ConnectedGraphLevel,GridGraphLevel}, # TargetLevel would be very expensive
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
    return l => (vʳ, λ, vˡ)
end

function allocate_intermediate(::EigMax, cgi::ConnectedGraphInit)
    (; C, W, Z_full) = cgi
    m = length(targetids(cgi))
    targetnodes = map(x -> x.node, targetids(cgi))
    nontargetnodes = setdiff(1:m, targetnodes)
    Mtarget = zeros(m, m)
    Mnontarget = zeros(length(nontargetnodes), m)
    return (; Mtarget, Mnontarget, targetnodes, nontargetnodes)
end

function compute_target(::EigMax, ti::TargetInit{<:Union{RSP,RandomWalk}})
    (; K, M, Mtarget, Mnontarget, targetnodes, nontargetnodes, Z_full) = ti
    idx = targetconnectedgraphidx(ti)

    Mtarget[:, idx] .= view(M, targetnodes)

    if size(Mnontarget, 1) > 0
        Mnontarget[:, idx] .= view(M, nontargetnodes)
    end
end

update_connectedgraph_output!(output, ::Level, ::EigMax, ::TargetInit, v) = nothing

# We do most of eigmax in finalize_connectedgraph_output!
function finalize_connectedgraph_output!(
    (vˡ, λ, vʳ), ::ConnectedGraphLevel, em::EigMax, cgi::ConnectedGraphInit, intermediates
)
    (; Mtarget, Mnontarget, targetnodes, nontargetnodes) = intermediates


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

    return vˡ, λ[], vʳ
end

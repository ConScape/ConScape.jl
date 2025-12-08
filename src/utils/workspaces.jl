
"""
    Workspaces(length::Int, n::Int)
    Workspaces(workspaces::Vector)

Workspaces aim to avoid repeated allocations of similar arrays.

Create a workspace of `n` `Vector{Float64}` of length `length`,
or pass in a custom Vector of Vectors.

Workspace vectors can be retrieved with `take!` and returned with `put!`.

The whole workspace can be free'd with `free!`, 
and all workspace arrays can be resized at once with `resize!(workspace, length)`.

Iterating over a `Workspaces` object `take!`s the next unused workspace(s).

```juila
ws = Workspaces(100, 5)
workspace1, workspace2 = ws
workspace3 = take!(ws)
# And return two of them.
put!(ws, workspace1)
put!(ws, workspace3)
````
"""
struct Workspaces{W,N}
    size::NTuple{N,Int}
    workspaces::Vector{W}
    unused::BitVector
end
Workspaces(size::Tuple, workspaces::Vector) = Workspaces(size, workspaces, trues(length(workspaces)))
Workspaces(size::Union{Int,Tuple}, n::Int) = Workspaces(Float64, size, n)
Workspaces(::Type{T}, length::Int, n::Int) where T = Workspaces(T, (length,), n)
Workspaces(::Type{T}, size::NTuple{N,Int}, n::Int) where {T,N} = 
    Workspaces(size, Array{T,N}[Array{T}(undef, size...) for _ in 1:n])

function Base.take!(ws::Workspaces{T})::T where T
    # Find the first unused workspace
    i = findfirst(ws.unused)
    # Error if there are no unused workspaces
    isnothing(i) && error("Not enough workspaces, used all $(length(ws.unused))")
    # Mark it as used
    ws.unused[i] = false
    # Return the workspace
    return ws.workspaces[i]
end
function Base.put!(ws::Workspaces, w)
    # Find the which workspace `w` was
    i = findfirst(x -> x === w, ws.workspaces)
    # Error if its not actually a workspace
    isnothing(i) && error("Workspace not found")
    # Mark it as unused
    ws.unused[i] = true
    return nothing
end
Base.iterate(ws::Workspaces, args...) = take!(ws), nothing
Base.length(ws::Workspaces) = length(first(ws.workspaces))
Base.size(ws::Workspaces) = size(first(ws.workspaces))

function Base.resize!(ws::Workspaces{V}, len::Int)::Workspaces{V} where {V<:AbstractVector}
    ws = map(w -> resize!(w, len), ws.workspaces)
    return Workspaces((len,), ws)
end
function Base.resize!(ws::Workspaces{A}, sze::Tuple)::Workspaces{A} where {A<:AbstractArray}
    len = prod(sze)
    ws = map(ws.workspaces) do w
        reshape(resize!(vec(w), len), sze)
    end 
    return Workspaces(sze, ws)
end

@noinline _not_matching_error(ws, sze) =
    error("Not matching $(length(ws)), $sze, $(map(size, ws.workspaces))")

free!(ws::Workspaces) = (ws.unused .= true; ws)

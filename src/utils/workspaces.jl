
"""
    Workspaces(length::Int, n::Int)
    Workspaces(size::Tuple, n::Int)
    Workspaces(template::SparseMatrixCSC, n::Int)

A pool of reusable arrays to avoid repeated allocations.

## Constructors
- `Workspaces(length, n)`: creates `n` vectors of given length
- `Workspaces(size, n)`: creates `n` matrices of given size
- `Workspaces(template, n)`: creates `n` sparse matrices matching template's pattern

## Usage
- `take!(ws)`: get an unused workspace (errors if none available)
- `put!(ws, w)`: return a workspace to the pool
- `free!(ws)`: mark all workspaces as unused
- `resize!(ws, len)`: resize all workspaces

## Example
```julia
ws = Workspaces(100, 5)
workspace1, workspace2 = ws  # iteration calls take!
workspace3 = take!(ws)
put!(ws, workspace1)         # return to pool
put!(ws, workspace3)
free!(ws)                    # mark all as unused
```
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
Workspaces(template::SparseMatrixCSC{Tv,Ti}, n::Int) where {Tv,Ti} =
    Workspaces(size(template), SparseMatrixCSC{Tv,Ti}[copy(template) for _ in 1:n])

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
Base.put!(ws::Workspaces, w::AbstractArray) = put!(ws, parent(w))
function Base.put!(ws::Workspaces, w::Array)
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
    for (i, w) in enumerate(ws.workspaces)
        ws.workspaces[i] = resize!(w, len)
    end 
    return free!(Workspaces((len,), ws.workspaces, ws.unused))
end
function Base.resize!(ws::Workspaces{A}, sze::Tuple)::Workspaces{A} where {A<:AbstractArray}
    len = prod(sze)
    for (i, w) in enumerate(ws.workspaces)
        ws.workspaces[i] = reshape(resize!(vec(w), len), sze)
    end 
    return free!(Workspaces(sze, ws.workspaces, ws.unused))
end
function Base.resize!(ws::Workspaces{A}, sze::Tuple{Int,Int})::Workspaces{A} where {A<:SparseMatrixCSC}
    for (i, w) in enumerate(ws.workspaces)
        ws.workspaces[i] = sparse!(w.csccolptr, w.cscrowval, w.nzvals, sze...) 
    end 
    return free!(Workspaces(sze, ws.workspaces, ws.unused))
end

@noinline _not_matching_error(ws, sze) =
    error("Not matching $(length(ws)), $sze, $(map(size, ws.workspaces))")

"""
    free!(ws::Workspaces)
    free!(wc::WorkspaceCollection)

Mark all workspaces as unused/available for `take!`.

## When to call free!

- After `_allocate_workspaces!`: marks fresh/resized workspaces as available
- At start of `_solve_connectedgraph!`: resets all workspaces for a new solve
- In `TargetInit`: frees only vec_workspaces between targets (mat/sp persist)

Note: `free!` does NOT clear the data in workspaces - it only marks them as
available. The data may be overwritten by the next `take!` user.
"""
free!(ws::Workspaces) = (ws.unused .= true; ws)

"""
    WorkspaceCollection

A container for all workspace types used in ConScape computations.

## Fields
- `vec`: Vector workspaces for target-level computations
- `mat`: Dense matrix workspaces for full Z/Y matrices
- `sp`: Sparse matrix workspaces for P, W, IW, etc.

## Lifecycle
1. Allocated via `_allocate_workspaces!` during GridGraphInit creation
2. Resized via `_allocate_workspaces!` for each ConnectedGraph
3. `free!(wc)` called at start of `_solve_connectedgraph!` to reset all
4. `free!(vec_workspaces(wc))` called between targets in TargetInit
5. Sparse workspaces persist through solve (used by precalculation results)

## Accessors
- `vec_workspaces(wc)`, `mat_workspaces(wc)`, `sp_workspaces(wc)`: get pool
- `vec_workspace(wc)`, `mat_workspace(wc)`, `sp_workspace(wc)`: take! one workspace
"""
struct WorkspaceCollection
    vec::Workspaces{Vector{Float64},1}
    mat::Workspaces{Matrix{Float64},2}
    sp::Workspaces{SparseMatrixCSC{Float64,Int64},2}
end

vec_workspaces(wc::WorkspaceCollection) = wc.vec
mat_workspaces(wc::WorkspaceCollection) = wc.mat
sp_workspaces(wc::WorkspaceCollection) = wc.sp

vec_workspace(wc::WorkspaceCollection) = take!(vec_workspaces(wc))
mat_workspace(wc::WorkspaceCollection) = take!(mat_workspaces(wc))
sp_workspace(wc::WorkspaceCollection) = take!(sp_workspaces(wc))

function free!(wc::WorkspaceCollection)
    free!(wc.vec)
    free!(wc.mat)
    free!(wc.sp)
    return wc
end


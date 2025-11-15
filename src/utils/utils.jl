
"""
    mapnz(f, A::SparseMatrixCSC)::SparseMatrixCSC

Map the non-zero values of a sparse matrix `A` with the function `f`.
"""
function mapnz(f, A::SparseMatrixCSC)
    B = copy(A)
    map!(f, B.nzval, A.nzval)
    return B
end
function mapnz(f, A::AbstractArray)
    B = copy(A)
    map!(f, B.data, A.data)
    return B
end

_prepare_qualities(A::AbstractMatrix) = _no_nan_f64.(_unwrap_raster(A))
_no_nan_f64(x) = Float64(x) # == isnan(x) ? 0.0 : Float64(x)
_unwrap_raster(R::Raster) = parent(R)
_unwrap_raster(R::AbstractMatrix) = R

function _fill_matrix(values, g::Initialisation)
    matrix = fill(NaN, size(g))
    matrix[sourceids(g)] .= values
    return matrix
end

function Raster(values::AbstractVector, p::Initialisation; kw...)
    ds = dims(p)
    isnothing(ds) && throw(ArgumentError("dims are `nothing` - it was not initialised with a Raster"))
    return Raster(_fill_matrix(values, p), ds::Tuple; kw...)
end

# function outdegrees(p::Initialisation)
#     values = sum(affinitymatrix(p), dims=2)
#     _maybe_raster(_fill_matrix(values, p), p)
# end

# function indegrees(p::Initialisation; kwargs...)
#     g = grid(p)
#     values = sum(affinitymatrix(g), dims=1)
#     _maybe_raster(_fill_matrix(values, p), p)
# end

# Prevent rewrapping
readonlyarray(A::AbstractArray) = ReadOnlyArray(A)
readonlyarray(A::ReadOnlyArray) = A

_maybe_set_diagonal!(ti::TargetInit, proximities) =
    _maybe_set_diagonal!(ti, proximities, diagvalue(ti))
_maybe_set_diagonal!(ti::TargetInit, proximities, diagvalue::Nothing) = proximities
function _maybe_set_diagonal!(ti::TargetInit, proximities, diagvalue::Number)
    proximities = ti.workspace .= proximities
    proximities[target(ti).node] = diagvalue
    return readonlyarray(proximities)
end
# function _maybe_set_diagonal!(proximitymatrix, diagvalue::Number, targetnodes::AbstractVector)
# , diagvalue(ti), target(ti).node
#     for (j, i) in enumerate(targetnodes)
#         proximitymatrix[i, j] = diagvalue
#     end
# end

# Fill a vector with zeros, and one for the target node
# If it was part of a square matrix this would be the diagonal
function _diag_vec!(workspace, target::TargetID)
    fill!(workspace, 0.0)
    workspace[target.node] = 1.0
    return workspace
end

# Reshape arrays to a new size dstructively
# This only makes sense if arrays are sorted large to small
function _reshape!(A::Array, size::Tuple{Vararg{Int}})
    len = prod(size)
    if Base.size(A) == size
        A
    else # if length(A) >= len
        # TODO make sure this doesn't allocate when the array is larger
        # We may need julia 1.11 to do this properly
        v = vec(A)::Vector
        resize!(v, len)
        reshape(v, size)
    end
end

_allocate_workspaces!(x, problem::ConScapeProblem, graph::ConnectedGraph) =
    _allocate_workspaces!(x, problem, nsources(graph))
_allocate_workspaces!(x::Nothing, problem::ConScapeProblem, length::Int) =
    Workspaces(length, nworkspaces(problem) + 20)
_allocate_workspaces!(workspaces::Workspaces, ::ConScapeProblem, length::Int) =
    free!(resize!(workspaces, length))

_maybe_new_outputs(level::GridGraphLevel, mes, ggi::GridGraphInit) =
    mes === measures(ggi) ? outputs(ggi) : allocate_output(level, mes, gridgraph(ggi), connectedgraphs(ggi))
_maybe_new_outputs(level::Level, mes, cgi::Union{ConnectedGraphInit,TargetInit}) =
    mes === measures(cgi) ? outputs(cgi) : allocate_output(level, mes, gridgraph(cgi), connectedgraph(cgi), precalculation(cgi))

# Get layers from a RasterStack or return nothing
_get_sourcequality(rast::RasterStack) = _keys_or_nothing(rast, (:sourcequality, :quality))
_get_targetquality(rast::RasterStack) = _keys_or_nothing(rast, (:targetquality, :quality, :sourcequality))
_get_likelihood(rast::RasterStack) = _keys_or_nothing(rast, (:likelihood, :steplikelihood))
_get_cost(rast::RasterStack) = _keys_or_nothing(rast, (:cost, :stepcost))

@inline _keys_or_nothing(rast, (key, keys...)::Tuple) =
    haskey(rast, key) ? rast[key] : _keys_or_nothing(rast, keys)
@inline _keys_or_nothing(rast, ::Tuple{}) = nothing

# Fast sparse array update, we loop over non-zero values and indices directly.
# adapted from `SparseArrays.findnz`
# This is painfully slow without this optimization
function foreachnz(f, S) 
    count = 1
    for col in 1:size(S, 2), k in SparseArrays.getcolptr(S)[col]:(SparseArrays.getcolptr(S)[col + 1] - 1)
        @inbounds i = SparseArrays.rowvals(S)[k]
        j = col
        f(i, j, count)
        count += 1
    end
    return nothing
end

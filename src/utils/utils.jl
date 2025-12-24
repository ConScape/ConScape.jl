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

_maybe_set_diagonal!(proximities::VDe, ti::TargetInit) =
    _maybe_set_diagonal!(proximities, ti, diagvalue(ti))
_maybe_set_diagonal!(proximities::VDe, ti::TargetInit, diagvalue::Nothing) = proximities
function _maybe_set_diagonal!(proximities::VDe, ti::TargetInit, diagvalue::Number)
    proximities[targetnode(ti)] = diagvalue
    return proximities
end
# function _maybe_set_diagonal!(proximitymatrix, diagvalue::Number, targetnodes::AbstractVector)
# , diagvalue(ti), targetnode(ti).node
#     for (j, i) in enumerate(targetnodes)
#         proximitymatrix[i, j] = diagvalue
#     end
# end

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

# Allocate or resize a WorkspaceCollection
# Connected graphs are sorted by size, so first is largest
function _allocate_workspaces!(wc, problem::ConScapeProblem, connectedgraphs::Vector)
    if length(connectedgraphs) > 0
        _allocate_workspaces!(wc, problem, first(connectedgraphs))
    elseif isnothing(wc)
        # Empty workspace collection
        WorkspaceCollection(
            Workspaces(0, 0),
            Workspaces((0, 0), 0),
            Workspaces(spzeros(0, 0), 0),
        )
    else
        wc
    end
end

function _allocate_workspaces!(wc::Nothing, problem::ConScapeProblem, graph::ConnectedGraph)
    len = nsources(graph)
    size = connectedgraph_size(graph)
    sp_template = steplikelihood(graph)
    sp_template = isnothing(sp_template) ? spzeros(len, len) : sp_template
    # Create fresh workspaces with required counts
    free!(WorkspaceCollection(
        Workspaces(len, num_vec_workspaces(problem)),
        Workspaces(size, num_mat_workspaces(problem)),
        Workspaces(sp_template, num_sp_workspaces(problem)),
    ))
end

function _allocate_workspaces!(wc::WorkspaceCollection, problem::ConScapeProblem, graph::ConnectedGraph)
    len = nsources(graph)
    size = connectedgraph_size(graph)
    sp_template = steplikelihood(graph)

    # Resize existing workspaces
    vec_ws = resize!(vec_workspaces(wc), len)
    mat_ws = resize!(mat_workspaces(wc), size)
    sp_ws = _update_sparse_template!(sp_workspaces(wc), sp_template)

    # Add more workspaces if needed
    needed_vec = num_vec_workspaces(problem)
    needed_mat = num_mat_workspaces(problem)
    needed_sp = num_sp_workspaces(problem)

    while length(vec_ws.workspaces) < needed_vec
        push!(vec_ws.workspaces, copy(first(vec_ws.workspaces)))
        push!(vec_ws.unused, true)
    end
    while length(mat_ws.workspaces) < needed_mat
        push!(mat_ws.workspaces, copy(first(mat_ws.workspaces)))
        push!(mat_ws.unused, true)
    end
    while length(sp_ws.workspaces) < needed_sp
        push!(sp_ws.workspaces, copy(first(sp_ws.workspaces)))
        push!(sp_ws.unused, true)
    end

    return free!(WorkspaceCollection(vec_ws, mat_ws, sp_ws))
end

# Get layers from a RasterStack or return nothing
_get_sourcequality(rast::RasterStack) = _keys_or_nothing(rast, (:sourcequality, :quality))
_get_targetquality(rast::RasterStack) = _keys_or_nothing(rast, (:targetquality, :quality, :sourcequality))
_get_likelihood(rast::RasterStack) = _keys_or_nothing(rast, (:likelihood, :steplikelihood))
_get_cost(rast::RasterStack) = _keys_or_nothing(rast, (:cost, :stepcost))

@inline _keys_or_nothing(rast, (key, keys...)::Tuple) =
    haskey(rast, key) ? rast[key] : _keys_or_nothing(rast, keys)
@inline _keys_or_nothing(rast, ::Tuple{}) = nothing

# Return a RasterStack if all outputs are Raster
_maybe_rasterstack(ggi) = _maybe_rasterstack(measures_outputs(ggi), ggi)
function _maybe_rasterstack(measures_outputs, ggi)
    out = _maybe_raster(measures_outputs, ggi)
    if all(map(o -> o isa Raster, out))
        return RasterStack(out)
    else
        return out
    end
end

# Return a Raster where possible
function _maybe_raster(
    measures_outputs::Union{Tuple,NamedTuple}, 
    g::Initialisation
)
    map(measures_outputs) do (; measure, output)
        _maybe_raster(returntrait(measure), output, dims(g); name=Symbol(measure))
    end
end
_maybe_raster(rt, x::Pair, g::Initialisation; kw...) = _maybe_raster(rt, x[1], g; kw...)
_maybe_raster(rt, x::Pair, g::Union{Tuple,Nothing}; kw...) = _maybe_raster(rt, x[1], g; kw...)
_maybe_raster(rt, rast::Raster, g::Initialisation; kw...) = rast
_maybe_raster(rt, mat::AbstractMatrix, g::Initialisation; kw...) =
    _maybe_raster(rt, mat, dims(g); kw...)
_maybe_raster(rt::ReturnSpatial, mat::Matrix{T}, dims::Tuple; kw...) where T =
    Raster(mat, dims; missingval=T(NaN), kw...)
_maybe_raster(rt::ReturnSpatial, vec::Vector{T}, dims::Tuple; kw...) where T<:Number =
    Raster(vec, dims; missingval=T(NaN), kw...)
_maybe_raster(rt, x, y; kw...) = x

_issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)
function _issquare(A::Union{TargetInit,ConnectedGraphInit}) 
    (a, b) = connectedgraph_size(A) 
    return a == b
end

function _split_by_level(cgi::ConnectedGraphInit)
    tlevel, cglevel = _split_by_level(measures_outputs(cgi))

    return setmeasures(cgi, tlevel), setmeasures(cgi, cglevel)
end
function _split_by_level(mos::NamedTuple{names}, ) where names
    # We need to wrap and unwrap `names` in `Val` so they 
    # stay in the type domain and dont lose type stability
    vnames = map(n -> Val{n}(), names)
    key_mo_tuple = map(Pair, vnames, values(mos))
    tkeys, cgkeys = reduce(key_mo_tuple; init=((), ())) do (t, cg), (k, v)  
        if computelevel(v.measure) isa TargetLevel
            ((t..., k), cg) # Add key to target keys 
        else
            (t, (cg..., k)) # Add key to connected graph keys
        end
    end

    return mos[map(_unwrap, tkeys)], mos[map(_unwrap, cgkeys)]
end

_unwrap(::Val{X}) where X = X

const MeasureOutputNamedTuple = NamedTuple{<:Any,<:Tuple{Vararg{MeasureOutput}}} 

# Update measures in and object.
# This lets us specify different measures after defining a problem.
@stable setmeasures(p::ConScapeProblem, m::Measure) =
    setmeasures(p, NamedTuple{(Symbol(m),)}((m,)))
@stable function setmeasures(p::ConScapeProblem, measures::Union{Tuple,NamedTuple})
    ConstructionBase.setproperties(p, (; measures))
end
@stable function setmeasures(ggi::GridGraphInit, m::NamedTuple{<:Any,Tuple{Vararg{MeasureOutput}}};
    finallevel=defaultfinallevel(ggi)
)
    problem = setmeasures(ConScape.problem(ggi), m)
    return ConstructionBase.setproperties(ggi, (; problem, outputs))
end
@stable setmeasures(ggi::GridGraphInit, m::Measure) =
    setmeasures(ggi, NamedTuple{(Symbol(m),)}((m,)))
@stable function setmeasures(ggi::GridGraphInit, m::MeasureNamedTuple)
    problem = setmeasures(ConScape.problem(ggi), m)
    outputs = map(measures(problem)) do m
        allocate_gridgraph_output(m, ggi)
    end
    return ConstructionBase.setproperties(ggi, (; problem, outputs))
end
@stable function setmeasures(cgi::ConnectedGraphInit, m::MeasureNamedTuple;
    finallevel=defaultfinallevel(cgi)
)
    problem = setmeasures(ConScape.problem(cgi), m)
    outputs = map(measures(problem)) do m
        allocate_connectedgraph_output(finallevel, m, cgi)
    end
    return ConnectedGraphInit(
        problem,
        gridgraph(cgi),
        connectedgraph(cgi),
        outputs,
        workspaces(cgi),
        storage(cgi),
        precalculation(cgi),
        connectedgraphid(cgi),
    )
end
@stable function setmeasures(cgi::ConnectedGraphInit, mos::MeasureOutputNamedTuple;
    finallevel=defaultfinallevel(cgi)
)
    return ConnectedGraphInit(
        problem(cgi),
        gridgraph(cgi),
        connectedgraph(cgi),
        mos,
        workspaces(cgi),
        storage(cgi),
        precalculation(cgi),
        connectedgraphid(cgi),
    )
end
@stable function setmeasures(ti::TargetInit, m; finallevel=defaultfinallevel(ti))
    connectedgraphinit = setmeasures(connectedgraphinit(ti), m; finallevel)
    return TargetInit(connectedgraphinit, target(ti))
end

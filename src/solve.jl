
# solve methods

# These all simply allocate an Initialisation objection then call `solve!`

solve(p::ConScapeProblem, x::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve!(init(p, x, args...; kw...))
# TODO: put measures into the ConScapeProblem before solving ?
solve(m::Union{MeasureTuple,MeasureNamedTuple}, p::ConScapeProblem, input::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve!(init(m, p, input, args...; kw...))
solve(m::Measure, p::ConScapeProblem, input::Union{GridGraph,RasterStack}, args...; kw...) = 
    only(solve!(init(m, p, input, args...; kw...)))
function solve(
    m::MeasureNamedTuple, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; kw...
) 
    solve!(init(m, movement, x, args...; kw...))
end
solve(m::Measure, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; kw...) =
    only(values(solve!(init(m, movement, x, args...; kw...))))
solve(::Initialisation, args...; kw...) = _solve_initialisation_error()
solve(a1::Union{ConScapeProblem,Measure,MeasureTuple,MeasureNamedTuple}, ::Initialisation, args...; kw...) = _solve_initialisation_error()

# solve! methods

# Allow solving all the levels of precalculated object with specific measures

solve!(i::Initialisation, m::Union{MeasureTuple,MeasureNamedTuple}, args...; kw...) =
    solve!(init(m, i, args...; kw...))
solve!(i::Initialisation, m::Measure, args...; kw...) =
    only(values(solve!(init(m, i, args...; kw...))))
solve!(ggi::GridGraphInit; kw...) = _solve_gridgraph!(ggi; kw...)
solve!(ggi::GridGraphInit, i::Int; kw...) = solve!(init(ggi, i; kw...))
solve!(ggi::GridGraphInit, i::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) =
    solve!(init(ggi, i, target; kw...))
solve!(cgi::ConnectedGraphInit; kw...) = _solve_connectedgraph!(cgi; kw...)
solve!(gi::ConnectedGraphInit, measures::Union{MeasureTuple,MeasureNamedTuple}, i::Int; kw...) =
    solve!(init(measures, gi, targetids(gi)[i]::typeof(TargetID)); kw...)
solve!(ti::TargetInit, measures::MeasureNamedTuple; finallevel=TargetLevel(), kw...) = 
    solve!(init(measures, ti; kw...); finallevel)
solve!(ti::TargetInit; finallevel=TargetLevel()) = _solve_target!(ti; finallevel)


# Actual solve Logic separated out from solve! dispatch

# Solve the whole grid into one output,
# possibly from multiple connected subgraphs
function _solve_gridgraph!(ggi; kw...)
    # Loop over subgraphs (there may be only one)
    for i in eachindex(connectedgraphs(ggi))
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        cgi = init(ggi, i)
        _solve_connectedgraph!(cgi)
        # Copy solved connected subgraphs to grid level outputs
        transfer_to_gridgraph_output!(outputs(ggi), cgi)
    end
    return _maybe_rasterstack(ggi)
end

# Solve a single connected graph, for all measures
function _solve_connectedgraph!(cgi; finallevel=ConnectedGraphLevel(), kw...)
    target_cgi, cg_cgi = _split_by_level(cgi)

    # ConnectedGraphLevel Measures
    # Solve the whole graph in one function
    map(measures_outputs(cg_cgi)) do (; measure, output)
        compute_connectedgraph!(output, measure, cg_cgi)
    end

    # TargetLevel Measures
    # Solve each target separately
    for target_id in targetids(target_cgi)
        # Precalculate for this target and graph measures
        targetinit = init(target_cgi, target_id)
        _solve_target!(targetinit; finallevel)
    end
    # Finalize output, where not all computations are target-by-target
    finalize_connectedgraph_output!(finallevel, target_cgi)

    return _maybe_rasterstack(cgi)
end

# Solve a single target pixel, for all measures computed at TargetLevel
@stable function _solve_target!(ti; finallevel)
    # Allocate target size outputs only?
    # This is only used when a single target is being solved.
    # outputs = if isnothing(ConScape.outputs(ti))
    #     allocate_output(finallevel, measures(ti), ti)
    # else
    #     ConScape.outputs(ti)
    # end

    # Compute for each measure and store outputs
    map(measures_outputs(ti)) do (; measure, output)
        ti_m = TargetInit(connectedgraphinit(ti), target(ti))
        # Dont compute the same measure multiple times
        store = storage(ti)
        key = Symbol(measure)
        if haskey(store, key)
            # We already have this variable, 
            update_connectedgraph_output!(output, finallevel, measure, ti_m, st[key])
        else
            result = compute_target!(output, finallevel, measure, ti_m)
            # Store simple array outputs
            if result isa AbstractVector
                store[key] = readonlyarray(result)
            end
        end
    end

    if finallevel isa TargetLevel
        # When just running one target we may return Raster/RasterStack
        return _maybe_rasterstack(measures, outputs, ti)
    else
        # Otherwise outputs as-is
        return outputs
    end
end

_solve_initialisation_error() = throw(ArgumentError("use solve! with Initialisation objects as the first argument"))

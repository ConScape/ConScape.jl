
# solve methods

# These allocate an Initialisation objection then call `solve!`

solve(p::ConScapeProblem, x::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve!(init(p, x, args...; kw...))
# TODO put measures into the ConScapeProblem before solving
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
solve!(ggi::GridGraphInit; kw...) = _solve_all_subgraphs!(ggi; kw...)
solve!(ggi::GridGraphInit, i::Int; kw...) = solve!(init(ggi, i; kw...))
solve!(ggi::GridGraphInit, i::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) =
    solve!(init(ggi, i, target; kw...))
solve!(cgi::ConnectedGraphInit; kw...) = _solve_all_targets!(cgi; kw...)
solve!(gi::ConnectedGraphInit, measures::Union{MeasureTuple,MeasureNamedTuple}, i::Int; kw...) =
    solve!(init(measures, gi, targetids(gi)[i]::typeof(TargetID)); kw...)
solve!(ti::TargetInit, measures::MeasureNamedTuple; outputlevel=TargetLevel(), kw...) = 
    solve!(init(measures, ti; kw...); outputlevel)
solve!(ti::TargetInit; outputlevel=TargetLevel()) = _solve_single_target!(ti, outputlevel)

# Logic separated out from solve! dispatch

function _solve_all_subgraphs!(ggi; kw...)
    # Loop over subgraphs (there may be only one)
    for i in eachindex(connectedgraphs(ggi))
        sgi = init(ggi, i)
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        solve!(sgi)
        # Copy solved connected subgraphs to grid level outputs
        transfer_to_gridgraph_output!(outputs(ggi), sgi)
    end
    return _maybe_rasterstack(ggi)
end

function _solve_all_targets!(cgi; kw...)
    # In case some measures need intermediate storage 
    # oar precalculations for all targets
    intermediates = map(measures(cgi)) do measure
        allocate_intermediate(measure, cgi)
    end
    # Solve each target separately
    for target_id in targetids(cgi)
        # Precalculate for this target and graph measures
        targetinit = init(cgi, target_id; intermediates)
        solve!(targetinit)
    end
    # Finalize output, where not all computations are target-by-target
    finalize_connectedgraph_output!(cgi, intermediates)
    return _maybe_rasterstack(cgi)
end

function _solve_single_target!(ti, outputlevel)
    # Allocate target vectors rather than matrices
    # TODO move this to the constructor?
    outputs1 = if isnothing(outputs(ti))
        allocate_output(outputlevel, measures(ti), ti)
    else
        outputs(ti)
    end
    # Store outputs
    results = map(outputs1, measures(ti), intermediates(ti)) do (output, level)::Pair{<:Any,<:Level}, measure, intermediate
        ti_m = TargetInit(connectedgraphinit(ti), target(ti), intermediate)
        # Dont compute the same measure multiple times
        st = storage(ti)
        x = Symbol(measure)
        if haskey(st, x)
            update_connectedgraph_output!(output, level, measure, ti_m, st[x])
        else
            val = compute_target!(output, level, measure, ti_m)
            # Store simple array outputs
            if val isa AbstractVector
                st[x] = readonlyarray(val)
            end
        end
    end
    # When just running one target we return Raster/RasterStack
    if outputlevel isa TargetLevel
        return _maybe_rasterstack(measures, outputs1, ti)
    else
        return results
    end
end

_solve_initialisation_error() = throw(ArgumentError("use solve! with Initialisation objects as the first argument"))

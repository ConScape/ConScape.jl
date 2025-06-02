
# solve methods

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

_solve_initialisation_error() = throw(ArgumentError("use solve! with Initialisation objects as the first argument"))

# Allow solving all the levels of precalculated object with specific measures
solve!(i::Initialisation, m::Union{MeasureTuple,MeasureNamedTuple}, args...; kw...) =
    solve!(init(m, i, args...; kw...))
solve!(i::Initialisation, m::Measure, args...; kw...) =
    only(values(solve!(init(m, i, args...; kw...))))
function solve!(ggi::GridGraphInit; kw...)
    # Loop over subgraphs (there may be only one)
    for i in eachindex(connectedgraphs(ggi))
        sgi = init(ggi, i)
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        solve!(sgi)
        transfer_output!(outputs(ggi), sgi)
    end
    return _maybe_rasterstack(ggi)
end
solve!(ggi::GridGraphInit, i::Int; kw...) = solve!(init(ggi, i; kw...))
solve!(ggi::GridGraphInit, i::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) =
    solve!(init(ggi, i, target; kw...))
function solve!(cgi::ConnectedGraphInit; kw...)
    for target_id in targetids(cgi)
        # Precalculate for this target and graph measures
        solve!(init(cgi, target_id))
    end
    finalize_output!(cgi)
    return _maybe_rasterstack(cgi)
end
solve!(gi::ConnectedGraphInit, measures::Union{MeasureTuple,MeasureNamedTuple}, i::Int; kw...) =
    solve!(init(measures, gi, targetids(gi)[i]::typeof(TargetID)); kw...)
solve!(ti::TargetInit, measures::MeasureNamedTuple; outputlevel=TargetLevel(), kw...) = 
    solve!(init(measures, ti; kw...); outputlevel)
function solve!(ti::TargetInit; outputlevel=TargetLevel())
    # Allocate target vectors rather than matrices
    # TODO move this to the constructor?
    outputs1 = if isnothing(outputs(ti))
        allocate_output(outputlevel, measures(ti), ti)
    else
        outputs(ti)
    end
    # Store outputs
    results = map(measures(ti), outputs1) do measure, output
        # Dont compute the same measure multiple times
        v = get_or_compute!(ti, measure)
        # Write values to output object
        update_output!(output, measure, ti, v)
    end
    # When just running one target we return Raster/RasterStack
    if outputlevel isa TargetLevel
        return _maybe_rasterstack(measures, outputs1, ti)
    else
        return results
    end
end

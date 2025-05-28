
# solve methods

solve(p::ConScapeProblem, x::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve(init(p, x, args...; kw...))
# TODO put measures into the ConScapeProblem before solving
solve(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, p::ConScapeProblem, input::Union{GridGraph,RasterStack}, args...; kw...) = 
    solve(init(m, p, input, args...; kw...))
function solve(
    m::MeasureNamedTuple, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; kw...
) 
    solve(init(m, movement, x, args...; kw...))
end
solve(m::Measure, movement::MovementMode, x::Union{GridGraph,RasterStack}, args...; kw...) =
    only(values(solve(init(m, movement, x, args...; kw...))))
# Allow solving all the levels of precalculated object with specific measures
solve(m::Union{MeasureTuple,MeasureNamedTuple}, i::Initialisation, args...; kw...) =
    solve(init(m, i, args...; kw...))
solve(m::Union{Measure,MeasureTuple,MeasureNamedTuple}, i::Initialisation, args...; kw...) =
    only(values(solve(init(m, i, args...; kw...))))
function solve(ggi::GridGraphInit; kw...)
    # Loop over subgraphs (there may be only one)
    for i in eachindex(connectedgraphs(ggi))
        sgi = init(ggi, i)
        # Intitalise sparse matrices and precalculate e.g. LU factorizations
        solve(sgi)
        transfer_output!(outputs(ggi), sgi)
    end
    return _maybe_rasterstack(ggi)
end
solve(ggi::GridGraphInit, i::Int; kw...) =
    solve(init(measures, ggi, i; outputlevel=ConnectedGraphLevel(), kw...))
solve(ggi::GridGraphInit, i::Int, target::Union{Int,CartesianIndex,TargetID}; kw...) =
    solve(init(measures, ggi, i, target; outputs=nothing, outputlevel=TargetLevel(), kw...))
function solve(cgi::ConnectedGraphInit; outputlevel=ConnectedGraphLevel(), kw...)
    for target_id in targetids(cgi)
        # Precalculate for this target and graph measures
        solve(init(cgi, target_id; kw...); outputlevel)
    end
    finalize_output!(cgi)
    return _maybe_rasterstack(cgi)
end
solve(measures::Union{MeasureTuple,MeasureNamedTuple}, gi::ConnectedGraphInit, i::Int; kw...) =
    solve(init(measures, gi, targetids(gi)[i]); kw...)
solve(measures::MeasureNamedTuple, ti::TargetInit; outputlevel=TargetLevel(), kw...) = 
    solve(init(measures, ti; kw...); outputlevel)
function solve(ti::TargetInit; outputlevel=TargetLevel())
    # Allocate target vectors rather than matrices
    outputs1 = if isnothing(outputs(ti))
        allocate_output(outputlevel, measures, ti)
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

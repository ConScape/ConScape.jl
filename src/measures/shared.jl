@inline function get_or_compute_target!(ti::TargetInit, m::Measure)
    st = storage(ti)
    x = Symbol(m)
    haskey(st, x) && return st[x]
    val = compute_target(m, ti)
    if returntrait(m) isa ReturnSpatial
        st[x] = readonlyarray(val)
        return readonlyarray(val)
    else
        return val
    end
end
@inline function get_or_compute_target!(ti::TargetInit, x::Symbol)
    st = storage(ti)
    haskey(st, x) && return st[x]

    val = target_precalculation!(ti, x)
    # TODO: storing other types
    if val isa Vector
        st[x] = readonlyarray(val)
    end
    return val
end

# Most measures dont need to deal with `update_connectedgraph_output` and just use `compute`
function compute_target!(output, l::Level, m::Measure, ti::TargetInit)
    val = compute_target(m, ti)
    update_connectedgraph_output!(output, l, m, ti, val)

    return output
end

#-------------------------------------------------------------------------------------------
# ConnectedGraph level measures

# Compute Target level measures for a connected subgraph
# function compute_target(m::Measure, cgi::ConnectedGraphInit)
#     output = allocate_output(
#     ConnectedGraphLevel(), m, problem(cgi), gridgraph(cgi), connectedgraph(cgi), precalculation(cgi)
#     )
#     for target in targetids(cgi)
#         ti = TargetInit(cgi, target)
#         compute_target!(output, m, ti)
#     end
#     return finalize_connectedgraph_output!(output, m, cgi)
# end


# What are these, how are they different to the RSP versions?
# compute_target(::ExpectedCost{BellmanFord}, ti::TargetInit{<:RSP}) = first(bellman_ford(ti))
# compute_target(::FreeEnergyDistance{BellmanFord}, ti::TargetInit{<:RSP}) = last(bellman_ford(ti))

# bellman_ford(ti::TargetInit{<:RSP}) =
    # first(bellman_ford(probabilitymatrix(ti), costmatrix(ti), theta(ti), target_id(ti), approx(ti)))

# TODO: handle self connectivity for single isolated nodes
# fill_isolated_node(::FunctionalHabitat, init::Initalisation, target::CartesianIndex) =
#      diagvalue(init) * sourcequality_spatial(init)[target] * targetquality_spatial(init)[target]
# fill_isolated_node(::Betweenness, init::Initalisation, target::CartesianIndex) = 0.0

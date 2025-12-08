@inline function get_or_compute_target!(ti::TargetInit, m::Measure)
    store = storage(ti)
    # TODO: this loses const propagation type stability
    x = Symbol(m)
    haskey(store, x) && return store[x]
    val = compute_target(m, ti)

    # For type stability only Vector{Float64} go in store
    if val isa Vector{Float64}
        store[x] = readonlyarray(val)
        return readonlyarray(val)
    else
        return val
    end
end
@inline function get_or_compute_target!(ti::TargetInit, x::Symbol)
    store = storage(ti)
    haskey(store, x) && return store[x]

    val = target_precalculation!(ti, x)
    # TODO: storing other types
    if val isa Vector{Float64}
        store[x] = readonlyarray(val)
    end

    if val isa Array
        return readonlyarray(val)
    else
        return val
    end
end

# Most measures dont need to deal with
# `update_connectedgraph_output!` and just define `compute_target`
function compute_target!(output, l::Level, m::Measure, ti::TargetInit)
    val = compute_target(m, ti)
    update_connectedgraph_output!(output, l, m, ti, val)

    return output
end

@generated Base.Symbol(m::Measure) = QuoteNode(nameof(m))

num_matrix_workspaces(::Measure, ::MovementMode) = 0
function num_matrix_workspaces(problem::ConScapeProblem)
    mes = measures(problem)
    mov = movement(problem)

    max_workspaces =
        mapreduce(m -> num_matrix_workspaces(m, mov), max, mes; init=0) +
        anymeasure(needs_full_fundamentalmatrix, mes, mov) +
        anymeasure(needs_full_fundamentalmatrix, mes, mov) +
        anymeasure(needs_full_costdistancematrix, mes, mov) +
        2 * anymeasure(needs_sensitivity_precursors, mes, mov)

    return max_workspaces
end

# Define the default output Level for that initialisation object
defaultfinallevel(::GridGraphInit) = GridGraphLevel()
defaultfinallevel(::ConnectedGraphInit) = ConnectedGraphLevel()
defaultfinallevel(::TargetInit) = TargetLevel()

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

"""
    sensitivity_simulation(problem, gridgraph; kw...)

Compute sensitivity of all nodes using finite-difference simulation.
This perturbs each edge/node and measures the change in the landscape metric.

This is used to validate the analytical sensitivity calculations.

## Keywords

- `wrt`: The input type to perturb. One of:
    - `StepLikelihood()` - perturb step likelihoods (was "A" for affinity)
    - `StepCost()` - perturb step costs (was "C")
    - `Quality()` - perturb quality values (was "Q")
    - `StepCostToLikelihood()` - perturb costs with linked likelihoods (was "C&A=f(C)")
    - `StepLikelihoodToCost()` - perturb likelihoods with linked costs (was "A&C=f(A)")
- `metric`: How to summarize the landscape matrix:
    - `Summation()` - sum of functional habitat
    - `EigMax()` - leading eigenvalue
- `type`: Whether to return sensitivity or elasticity:
    - `Sensitivity()` - raw sensitivity
    - `Elasticity()` - proportional/unitless sensitivity
- `epsi`: Perturbation size for finite difference (default: 1e-6)
- `one_out_of`: Subsampling factor to speed up computation (default: 1, meaning all pixels)
"""
function sensitivity_simulation(
    problem::ConScapeProblem,
    gridgraph::ConScape.GridGraph;
    wrt::ConScape.InputType=StepLikelihood(),
    metric::Union{Summation,EigMax}=Summation(),
    type::ConScape.SensitivityType=Sensitivity(),
    epsi::Float64=1e-6,
    one_out_of::Int=1,
)
    # Get the base landscape metric value
    base_metric = _compute_landscape_metric(problem, gridgraph, metric)

    n = ConScape.nsources(gridgraph)

    if wrt isa ConScape.Permeability
        # Edge-based sensitivity (StepLikelihood, StepCost, or linked)
        edge_sensitivities = _edge_sensitivity_simulation(
            problem, gridgraph, wrt, metric, type, base_metric, epsi, one_out_of
        )
        # Sum edge sensitivities to get node sensitivities
        # dims=1 sums incoming edges to each node (column sums)
        # This matches the analytical code which adds edge (i,j) sensitivity to node j
        node_sensitivities = vec(sum(edge_sensitivities; dims=1))
    else
        # Node-based sensitivity (Quality)
        node_sensitivities = _quality_sensitivity_simulation(
            problem, gridgraph, wrt, metric, type, base_metric, epsi, one_out_of
        )
    end

    # Map back to spatial grid
    return _to_spatial(gridgraph, node_sensitivities)
end

function _compute_landscape_metric(problem, gridgraph, ::Summation)
    measures = (; fh=FunctionalHabitat())
    result = solve(measures, problem, gridgraph)
    fh = result.fh
    replace!(x -> isnan(x) ? 0.0 : x, fh)
    return sum(fh)
end
function _compute_landscape_metric(problem, gridgraph, eigmax::EigMax)
    measures = (; eigmax)
    result = solve(measures, problem, gridgraph)
    # eigmax returns (vˡ, λ, vʳ), we want λ
    return result.eigmax[1][2][]  # First connected graph, second element (λ), unwrap Ref
end

function _edge_sensitivity_simulation(
    problem, gridgraph, wrt, metric, sensitivitytype, base_metric, epsi, one_out_of
)
    A = gridgraph.steplikelihood
    C = gridgraph.stepcost
    edge_sensitivities = copy(A)

    n = ConScape.nsources(gridgraph)

    for i in 1:n
        successors = findall(A[i, :] .> 0)
        for j in successors
            if j % one_out_of == 0
                # Create perturbed graph
                new_gridgraph = _perturb_edge(gridgraph, wrt, i, j, epsi)

                # Compute new metric
                new_metric = _compute_landscape_metric(problem, new_gridgraph, metric)

                # Finite difference
                sensitivity = (new_metric - base_metric) / epsi

                # Scale for elasticity if needed
                # Scale by the parameter being perturbed
                if sensitivitytype isa Elasticity
                    if wrt isa Union{StepLikelihood,StepCostToLikelihood}
                        # These perturb likelihood
                        sensitivity *= A[i, j]
                    else  # StepCost or StepLikelihoodToCost perturb cost
                        sensitivity *= C[i, j]
                    end
                end

                edge_sensitivities[i, j] = sensitivity
            else
                edge_sensitivities[i, j] = NaN
            end
        end
    end

    return edge_sensitivities
end

function _perturb_edge(gridgraph, ::StepLikelihood, i, j, epsi)
    new_likelihood = copy(gridgraph.steplikelihood)
    new_likelihood[i, j] += epsi
    return ConScape.GridGraph(;
        steplikelihood=new_likelihood,
        stepcost=gridgraph.stepcost,
        sourcequality=gridgraph.sourcequality,
        targetquality=gridgraph.targetquality,
    )
end
function _perturb_edge(gridgraph, ::StepCost, i, j, epsi)
    new_cost = copy(gridgraph.stepcost)
    new_cost[i, j] += epsi
    return ConScape.GridGraph(;
        steplikelihood=gridgraph.steplikelihood,
        stepcost=new_cost,
        sourcequality=gridgraph.sourcequality,
        targetquality=gridgraph.targetquality,
    )
end
# StepCostToLikelihood = "A&C=f(A)" = perturb likelihood, cost is derived
function _perturb_edge(gridgraph, ::StepCostToLikelihood, i, j, epsi)
    new_likelihood = copy(gridgraph.steplikelihood)
    new_likelihood[i, j] += epsi
    # Link cost to likelihood via MinusLog: C = -log(A)
    new_cost = ConScape.mapnz(MinusLog(), new_likelihood)
    return ConScape.GridGraph(;
        steplikelihood=new_likelihood,
        stepcost=new_cost,
        sourcequality=gridgraph.sourcequality,
        targetquality=gridgraph.targetquality,
    )
end
# StepLikelihoodToCost = "C&A=f(C)" = perturb cost, likelihood is derived
function _perturb_edge(gridgraph, ::StepLikelihoodToCost, i, j, epsi)
    new_cost = copy(gridgraph.stepcost)
    new_cost[i, j] += epsi
    # Link likelihood to cost via inverse MinusLog: A = exp(-C)
    new_likelihood = copy(new_cost)
    map!(x -> exp(-x), new_likelihood.nzval, new_cost.nzval)
    return ConScape.GridGraph(;
        steplikelihood=new_likelihood,
        stepcost=new_cost,
        sourcequality=gridgraph.sourcequality,
        targetquality=gridgraph.targetquality,
    )
end

function _quality_sensitivity_simulation(
    problem, gridgraph, wrt, metric, sensitivitytype, base_metric, epsi, one_out_of
)
    n = ConScape.nsources(gridgraph)
    node_sensitivities = zeros(n)

    source_ids = ConScape.sourceids(gridgraph)

    for (idx, spatial_idx) in enumerate(source_ids)
        if idx % one_out_of == 0
            # Perturb quality at this node
            new_gridgraph = _perturb_quality(gridgraph, wrt, spatial_idx, epsi)

            # Compute new metric
            new_metric = _compute_landscape_metric(problem, new_gridgraph, metric)

            # Finite difference
            sensitivity = (new_metric - base_metric) / epsi

            # Scale for elasticity if needed
            if sensitivitytype isa Elasticity
                if wrt isa Union{Quality,SourceQuality}
                    sensitivity *= gridgraph.sourcequality[spatial_idx]
                else  # TargetQuality
                    sensitivity *= gridgraph.targetquality[spatial_idx]
                end
            end

            node_sensitivities[idx] = sensitivity
        else
            node_sensitivities[idx] = NaN
        end
    end

    return node_sensitivities
end

function _perturb_quality(gridgraph, ::Quality, spatial_idx, epsi)
    new_sourcequality = copy(gridgraph.sourcequality)
    new_targetquality = copy(gridgraph.targetquality)
    new_sourcequality[spatial_idx] += epsi
    new_targetquality[spatial_idx] += epsi
    return ConScape.GridGraph(;
        steplikelihood=gridgraph.steplikelihood,
        stepcost=gridgraph.stepcost,
        sourcequality=new_sourcequality,
        targetquality=new_targetquality,
    )
end
function _perturb_quality(gridgraph, ::SourceQuality, spatial_idx, epsi)
    new_sourcequality = copy(gridgraph.sourcequality)
    new_sourcequality[spatial_idx] += epsi
    return ConScape.GridGraph(;
        steplikelihood=gridgraph.steplikelihood,
        stepcost=gridgraph.stepcost,
        sourcequality=new_sourcequality,
        targetquality=gridgraph.targetquality,
    )
end
function _perturb_quality(gridgraph, ::TargetQuality, spatial_idx, epsi)
    new_targetquality = copy(gridgraph.targetquality)
    new_targetquality[spatial_idx] += epsi
    return ConScape.GridGraph(;
        steplikelihood=gridgraph.steplikelihood,
        stepcost=gridgraph.stepcost,
        sourcequality=gridgraph.sourcequality,
        targetquality=new_targetquality,
    )
end

function _to_spatial(gridgraph, node_sensitivities)
    source_ids = ConScape.sourceids(gridgraph)
    nrows, ncols = size(gridgraph)

    result = fill(NaN, nrows, ncols)
    for (idx, spatial_idx) in enumerate(source_ids)
        result[spatial_idx] = node_sensitivities[idx]
    end

    # Mask by source quality (NaN where quality is NaN)
    result .*= map(x -> isnan(x) ? NaN : 1.0, gridgraph.sourcequality)

    return result
end

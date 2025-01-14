"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure end

keywords(o::GraphMeasure) = _keywords(o)

abstract type TopologicalMeasure <: GraphMeasure end
abstract type BetweennessMeasure <: GraphMeasure end
abstract type PerturbationMeasure <: GraphMeasure end
abstract type PathDistributionMeasure <: GraphMeasure end

struct BetweennessQweighted <: BetweennessMeasure end
@kwdef struct BetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end
struct EdgeBetweennessQweighted <: BetweennessMeasure end
@kwdef struct EdgeBetweennessKweighted{DV} <: BetweennessMeasure 
    diagvalue::DV=nothing
end

@kwdef struct ConnectedHabitat{DV} <: GraphMeasure
    diagvalue::DV=nothing
end

@kwdef struct Criticality{DV,AV,QT,QS} <: PerturbationMeasure 
    diagvalue::DV=nothing
    avalue::AV=floatmin()
    qˢvalue::QS=0.0
    qᵗvalue::QT=0.0
end

# These maybe don't quite belong here?
@kwdef struct EigMax{DV,T} <: TopologicalMeasure
    diagvalue::DV=nothing
    tol::T=1e-14
end

struct MeanLeastCostKullbackLeiblerDivergence <: PathDistributionMeasure end
struct MeanKullbackLeiblerDivergence <: PathDistributionMeasure end

# Map structs to functions

# These return Rasters
graph_function(m::BetweennessKweighted) = betweenness_kweighted
graph_function(m::BetweennessQweighted) = betweenness_qweighted
graph_function(m::ConnectedHabitat) = connected_habitat
graph_function(m::Criticality) = criticality
# These return scalars
graph_function(m::MeanLeastCostKullbackLeiblerDivergence) = mean_lc_kl_divergence
graph_function(m::MeanKullbackLeiblerDivergence) = mean_kl_divergence
# These return sparse arrays
graph_function(m::EdgeBetweennessKweighted) = edge_betweenness_kweighted
graph_function(m::EdgeBetweennessQweighted) = edge_betweenness_qweighted
# Returns a tuple
graph_function(m::EigMax) = eigmax

# Map structs to function keywords, 
# a bit of a hack until we refactor the rest
keywords(gm::GraphMeasure, p::AbstractProblem) = 
    (; _keywords(gm)...)#, solver=solver(p))
keywords(gm::ConnectedHabitat, p::AbstractProblem) = 
    (; _keywords(gm)..., approx=connectivity_measure(p).approx)#, solver=solver(p))

# A trait for connectivity requirement
struct NeedsConnectivity end
struct NoConnectivity end
needs_connectivity(::GraphMeasure) = NoConnectivity()
needs_connectivity(::BetweennessKweighted) = NeedsConnectivity()
needs_connectivity(::EdgeBetweennessKweighted) = NeedsConnectivity()
needs_connectivity(::EigMax) = NeedsConnectivity()
needs_connectivity(::ConnectedHabitat) = NeedsConnectivity()
needs_connectivity(::Criticality) = NeedsConnectivity()

# compute
# This is where things actually happen
#
# Add dispatch on connectivity measure
compute(gm::GraphMeasure, p::AbstractProblem, g::Union{Grid,GridRSP}; kw...) = 
    compute(needs_connectivity(gm), gm, p, g; kw...)
function compute(::NeedsConnectivity,
    gm::GraphMeasure, 
    p::AbstractProblem, 
    g::Union{Grid,GridRSP};
    workspace_kw...
)
    cm = p.connectivity_measure
    distance_transformation = cm.distance_transformation
    connectivity_function = ConScape.connectivity_function(cm)
    # Handle multiple distance transformations
    if distance_transformation isa NamedTuple
        map(distance_transformation) do dt
            graph_function(gm)(g; 
                keywords(gm, p)..., 
                distance_transformation=dt, 
                connectivity_function,
                workspace_kw...
            )
        end
    else
        graph_function(gm)(g; 
            keywords(gm, p)..., 
            distance_transformation=dt, 
            connectivity_function,
            workspace_kw...
        )
    end
end
function compute(::NoConnectivity,
    gm::GraphMeasure, 
    p::AbstractProblem, 
    g::Union{Grid,GridRSP}; 
    workspace_kw...
) 
    graph_function(gm)(g; keywords(gm, p)..., workspace_kw...)
end

# Workspace allocation traits
needs_inv(::GraphMeasure) = false
needs_inv(::BetweennessMeasure) = true
needs_workspace(::GraphMeasure) = false
needs_workspace(::BetweennessMeasure) = true
needs_expected_cost(::GraphMeasure) = false
needs_expected_cost(::EdgeBetweennessKweighted) = true
needs_expected_cost(::MeanKullbackLeiblerDivergence) = true
needs_free_energy_distance(::GraphMeasure) = false
needs_free_energy_distance(::MeanKullbackLeiblerDivergence) = true
needs_Aaj_init(::GraphMeasure) = true
hastrait(t, gms) = mapreduce(t, |, gms; init=false)

function _setup_workspace(p::AbstractProblem, grsp::GridRSP; kw...)
    gms = p.graph_measures
    workspace1 = if hastrait(needs_workspace, gms)
        similar(grsp.Z)
    else
        nothing
    end
    Zⁱ = if hastrait(needs_inv, gms)
        _inv(grsp.Z) 
    else
        nothing
    end
    Aadj_init, Aadj = if hastrait(needs_Aaj_init, gms)
        Aadj = (I - grsp.W)'
        solver_init(p.solver, Aadj), Aadj 
    else
        nothing
    end
    workspace_kw =  (; Zⁱ, workspace1, Aadj_init, Aadj)
    expected_cost = if hastrait(needs_expected_cost, gms)
        ConScape.expected_cost(grsp; workspace_kw..., kw...)
    else
        nothing
    end
    free_energy_distance = if hastrait(needs_free_energy_distance, gms)
        ConScape.free_energy_distance(grsp; workspace_kw..., kw...)
    else
        nothing
    end

    return (; workspace_kw..., kw..., expected_cost, free_energy_distance)
end
"""
    GraphMeasure 

Abstract supertype for graph measures.
These are lazy definitions of conscape functions.
"""
abstract type GraphMeasure end

abstract type ReturnType end
struct ReturnsDenseSpatial <: ReturnType end
struct ReturnsSparse <: ReturnType end
struct ReturnsScalar <: ReturnType end
struct ReturnsOther{F} <: ReturnType 
    f::F
end

"""
    NoWriteArray

A Julia AbstractArray wrapper that errors on `setindex!`, for testing.
"""
mutable struct NoWriteArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    __data::A
end

Base.size(A::NoWriteArray) = size(A.__data)
Base.copy(A::NoWriteArray) = copy(A.__data)
Base.getindex(A::NoWriteArray, i...) = A.__data[i...]
Base.setindex!(A::NoWriteArray, v, i...) = error("Cannot write to NoWriteArray")
Base.:(==)(A::NoWriteArray, B::NoWriteArray) = A.__data == B.__data

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

returntype(::EdgeBetweennessQweighted) = ReturnsSparse()
returntype(::EdgeBetweennessKweighted) = ReturnsSparse()
returntype(::BetweennessQweighted) = ReturnsDenseSpatial()
returntype(::BetweennessKweighted) = ReturnsDenseSpatial()
returntype(::ConnectedHabitat) = ReturnsDenseSpatial()
returntype(::Criticality) = ReturnsDenseSpatial()
returntype(::EigMax) = ReturnsOther((n, m) -> n + m)
returntype(::MeanLeastCostKullbackLeiblerDivergence) = ReturnsScalar()
returntype(::MeanKullbackLeiblerDivergence) = ReturnsScalar()

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
    (; _keywords(gm)..., solver=solver(p))
keywords(gm::ConnectedHabitat, p::AbstractProblem) = 
    (; _keywords(gm)..., approx=connectivity_measure(p).approx, solver=solver(p))

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
return_type(::GraphMeasure) = false
needs_inv(::GraphMeasure) = false
needs_inv(::BetweennessMeasure) = true
needs_workspaces(::GraphMeasure) = 0
needs_workspaces(::BetweennessMeasure) = 2
needs_workspaces(
    ::Union{EdgeBetweennessKweighted,EdgeBetweennessQweighted}
) = 3
needs_permuted_workspaces(::GraphMeasure) = 0
needs_permuted_workspaces(::EdgeBetweennessKweighted) = 1
needs_edge_betweennesses(::GraphMeasure) = false
needs_edge_betweennesses(
    ::Union{EdgeBetweennessKweighted,EdgeBetweennessQweighted}
) = true
needs_dense_A(::GraphMeasure) = false
needs_dense_A(
    ::Union{EdgeBetweennessKweighted,EdgeBetweennessQweighted}
) = true
needs_expected_cost(::GraphMeasure) = false
needs_expected_cost(::EdgeBetweennessKweighted) = true
needs_expected_cost(::MeanKullbackLeiblerDivergence) = true
needs_free_energy_distance(::GraphMeasure) = false
needs_free_energy_distance(::MeanKullbackLeiblerDivergence) = true
needs_Aaj_init(::GraphMeasure) = true
hastrait(t, gms) = mapreduce(t, |, gms; init=false)

function _measures_workspace(p::AbstractProblem, grsp::GridRSP; 
    A, 
    A_init,
    workspace,
    kw...
)
    gms = p.graph_measures
    n_workspaces = mapreduce(needs_workspaces, max, gms)
    n_permuted_workspaces = mapreduce(needs_permuted_workspaces, max, gms)
    workspaces = [workspace, (similar(grsp.Z) for _ in 1:n_workspaces-1)...]
    permuted_workspaces = [similar(grsp.Z') for _ in 1:n_permuted_workspaces]
    Zⁱ = if hastrait(needs_inv, gms)
        NoWriteArray(_inv(grsp.Z))
    else
        nothing
    end
    Aadj_init, Aadj = if hastrait(needs_Aaj_init, gms)
        # Just take the adjoint of the factorization of A
        # where possible to save calculations and memory
        Aadj_init, Aadj = if hasproperty(A_init, :F)
            Aadj = A'
            # Use adjoint factorization of A rather than recalculating for A'
            Aadj_init = merge(A_init, (; F=A_init.F'))
            Aadj_init, Aadj
        else
            # LinearSolve.jl cant handle the adjoint 
            # so we duplicate work and allocations
            Aadj = sparse(A')
            Aadj_init = init(solver(p), Aadj)
            Aadj_init, Aadj
        end
        Aadj_init, Aadj
    else
        nothing, nothing
    end
    # Create an intermediate workspace to use in computations
    workspace_kw =  (; Zⁱ, workspaces, permuted_workspaces, Aadj_init, Aadj, A, A_init, kw...)
    cf = connectivity_function(p)
    expected_costs = if hastrait(needs_expected_cost, gms) || cf == ConScape.expected_cost
        NoWriteArray(ConScape.expected_cost(grsp; workspace_kw..., solver=solver(p), kw...))
    else
        nothing
    end
    free_energy_distances = if hastrait(needs_free_energy_distance, gms) || cf == ConScape.free_energy_distance
        NoWriteArray(ConScape.free_energy_distance(grsp; workspace_kw..., solver=solver(p), kw...))
    else
        nothing
    end
    edge_betweennesses = if hastrait(needs_edge_betweennesses, gms)
        copy(grsp.W)
    else
        nothing
    end

    CW = grsp.g.costmatrix .* grsp.W
    return (; grsp, workspace_kw..., CW, free_energy_distances, expected_costs, edge_betweennesses)
end
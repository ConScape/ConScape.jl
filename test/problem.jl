using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL, Plots
using LinearSolve

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

mov_prob = replace_missing(Raster(joinpath(datadir, "mov_prob_1000.asc")), NaN)
hab_qual = replace_missing(Raster(joinpath(datadir, "hab_qual_1000.asc")), NaN)
mask!(mov_prob; with=hab_qual)
mask!(hab_qual; with=mov_prob)
rast = RasterStack((; affinities=mov_prob, qualities=hab_qual, target_qualities=hab_qual))
rast.qualities[(rast.affinities .> 0) .& isnan.(rast.qualities)] .= 1e-20
size(rast)
# rast = ConScape.coarse_graining(rast, 10)

graph_measures = graph_measures = (;
    func=ConScape.ConnectedHabitat(),
    qbetw=ConScape.BetweennessQweighted(),
    kbetw=ConScape.BetweennessKweighted(),
    # TODO sens=ConScape.Sensitivity(),
    # eigmax=ConScape.EigMax(),
    # qedgebetw=ConScape.EdgeBetweennessQweighted(),
    # kedgebetw=ConScape.EdgeBetweennessKweighted(),
    # mkld=ConScape.MeanKullbackLeiblerDivergence(),
    # mlcd=ConScape.MeanLeastCostKullbackLeiblerDivergence(),
    # crit=ConScape.Criticality(), # very very slow, each target makes a new grid
)
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor())
connectivity_measure = ConScape.ExpectedCost(; θ=1.0, distance_transformation)

expected_layers = (:func_exp, :func_oddsfor, :qbetw, :kbetw_exp, :kbetw_oddsfor, :mkld, :mlcd)

# Basic Problem
problem = ConScape.Problem(; 
    graph_measures, connectivity_measure, solver=ConScape.MatrixSolver(),
)
@time workspace = init(problem, rast; prune=true);
workspace.B_sparse
map(x -> x / 1e6, ConScape.allocations(problem, rast))
map(x -> x / 1e6, ConScape.allocations(problem, size(workspace.B_sparse)))

ConScape.allocations(problem, rast).total / 1e6
Base.summarysize(workspace) / 1e6
ConScape.allocations(problem, size(workspace.B_sparse)).total / 1e6

map(x -> Base.summarysize(x) / 1e6, workspace)
map(propertynames(workspace.grid)) do n
    n => Base.summarysize(getproperty(workspace.grid, n)) / 1e6
end

using BenchmarkTools
@time result = ConScape.solve(problem, workspace);
@btime result = ConScape.solve(problem, workspace);
# workspace_copy = deepcopy(workspace)
# workspace.expected_costs
# workspace_copy.expected_costs
# map(workspace, workspace_copy) do x, y
#     if x isa Union{Tuple,NamedTuple} 
#         all(map(==, x, y))
#     else
#         x == y
#     end
# end
using BenchmarkTools
@time workspace = init(problem, rast);
@btime ConScape.solve(problem, workspace);
@profview_allocs workspace = init(problem, rast)
@profview_allocs ConScape.solve(problem, workspace) sample_rate=1.0
#@profview_allocs ConScape.solve(problem, workspace)
@test result isa RasterStack
@test size(result) == size(rast)
@test keys(result) == expected_layers

plot(result)
sum(skipmissing(rebuild(result.func_exp; missingval=NaN)))
Base.summarysize(workspace) / 1e6

400 * 400 * 21 * 21 / 1e6 * sizeof(Float64) * 8
@profview ConScape.init(problem, rast)
@profview ConScape.solve(problem, rast; workspace)
ConScape.solve(problem, rast)

# Threaded solve problem
vector_problem = ConScape.Problem(; 
    graph_measures, connectivity_measure,
    solver = ConScape.VectorSolver(; threaded=true),
)
@time workspace = init(vector_problem, rast);
@time vector_result = ConScape.solve(vector_problem, workspace);
@btime vector_result = ConScape.solve(vector_problem, workspace);
@test vector_result isa RasterStack
@test size(vector_result) == size(rast)
@test keys(vector_result) == expected_layers
@test all(vector_result.func_exp .=== result.func_exp)
Plots.plot(vector_result)

Base.summarysize(workspace) / 1e6
sum(skipmissing(rebuild(vector_result.func_exp; missingval=NaN)))
@profview workspace = init(vector_problem, rast);
@profview ConScape.solve(vector_problem, workspace)
map(w -> Base.summarysize(w) / 10^6, workspace) 
map(w -> Base.summarysize(w) / 10^6, workspace.A_init) 

# Problem with custom solver
linearsolve_problem = ConScape.Problem(; 
    graph_measures, connectivity_measure,
    solver = ConScape.LinearSolver(MKLPardisoIterate(; nprocs=20)),
    # solver = ConScape.LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I))),
)
Base.summarysize(workspace) / 1e6
@time ls_result = ConScape.solve(linearsolve_problem, rast)
@test ls_result isa RasterStack
@test size(ls_result) == size(rast)
@test keys(ls_result) == expected_layers

@profview ConScape.init(linearsolve_problem, rast)
@profview ConScape.solve(linearsolve_problem, rast)

# WindowedProblem returns a RasterStack
windowed_problem = ConScape.WindowedProblem(problem; 
    radius=40, overlap=10, threaded=true
)
windowed_result = ConScape.solve(windowed_problem, rast, verbose=true)
plot(windowed_result)

using GLMakie
Rasters.rplot(windowed_result)
@test windowed_result isa RasterStack
@test size(windowed_result) == size(rast)
@test keys(windowed_result) == expected_layers 

window_tiles = ConScape.solve(windowed_problem, rast; test_windows=true, verbose=true)
plot(window_tiles)
Rasters.rplot(window_tiles)

# StoredProblem writes files to disk and mosaics to RasterStack

stored_problem = ConScape.StoredProblem(problem; 
    path=tempname(), radius=40, overlap=10, threaded=true
)
ConScape.solve(stored_problem, rast; verbose=true)
stored_result = mosaic(stored_problem; to=rast)
@test stored_result isa RasterStack
@test size(stored_result) == size(rast)
# keys are sorted now from file-name order
@test keys(stored_result) == Tuple(sort(collect(expected_layers)))
# Check the answer matches the WindowedProblem
@test all(stored_result.func_exp .=== windowed_result.func_exp)

plot(stored_result)
Rasters.rplot(stored_result.func_exp .- result.func_exp)
sum(skipmissing(windowed_result.func_exp))
sum(skipmissing(stored_result.func_exp))
sum(skipmissing(rebuild(result.func_exp; missingval=NaN)))

# StoredProblem can be run as batch jobs for clusters
# We just need a new path to make sure the result is from a new run
stored_problem2 = ConScape.StoredProblem(problem; 
    path=tempname(), radius=40, overlap=10, threaded=true
)
njobs = ConScape.count_batches(stored_problem2, rast) 
@test jobs isa Vector{Int}

for job in 1:njobs
    ConScape.solve(stored_problem2, rast, job)
end
batch_result = mosaic(stored_problem2; to=rast)
# Check the answer matches the non-batched run
@test all(batch_result.func_exp .=== stored_result.func_exp)
@test keys(batch_result) == Tuple(sort(collect(expected_layers)))

# StoredProblem can be nested with WindowedProblem
small_windowed_problem = ConScape.WindowedProblem(problem; 
    radius=25, overlap=10,
)
nested_problem = ConScape.StoredProblem(small_windowed_problem; 
    path=tempname(), radius=40, overlap=10, threaded=false
)
ConScape.solve(nested_problem, rast)
nested_result = mosaic(nested_problem; to=rast)
@test nested_result isa RasterStack
@test size(nested_result) == size(rast)
@test keys(nested_result) == Tuple(sort(collect(expected_layers)))
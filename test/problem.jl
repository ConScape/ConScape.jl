using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL, Plots
using LinearSolve

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

mov_prob = replace_missing(Raster(joinpath(datadir, "mov_prob_1000.asc")), NaN)
hab_qual = replace_missing(Raster(joinpath(datadir, "hab_qual_1000.asc")), NaN)
rast = RasterStack((; affinities=mov_prob, qualities=hab_qual, target_qualities=hab_qual))
rast.qualities[(rast.affinities .> 0) .& isnan.(rast.qualities)] .= 1e-20
#rast = ConScape.coarse_graining(rast, 10)


graph_measures = graph_measures = (;
    func=ConScape.ConnectedHabitat(),
    qbetw=ConScape.BetweennessQweighted(),
    kbetw=ConScape.BetweennessKweighted(),
    # mkld=ConScape.MeanKullbackLeiblerDivergence(),
    # mlcd=ConScape.MeanLeastCostKullbackLeiblerDivergence(),
)
distance_transformation = (exp=x -> exp(-x/75), oddsfor=ConScape.OddsFor())
connectivity_measure = ConScape.ExpectedCost(; θ=1.0, distance_transformation)

expected_layers = (:func_exp, :func_oddsfor, :qbetw, :kbetw_exp, :kbetw_oddsfor)#, :mkld, :mlcd)

# Basic Problem
problem = ConScape.Problem(; 
    graph_measures, connectivity_measure, solver=ConScape.MatrixSolver(),
)
@time workspace = init(problem, rast);
@time result = ConScape.solve(problem, rast; workspace)
@test result isa RasterStack
@test size(result) == size(rast)
@test keys(result) == expected_layers

plot(result)
map(Base.summarysize, workspace)
Base.summarysize(workspace)
@profview ConScape.init(problem, rast)
@profview ConScape.solve(problem, rast; workspace)
ConScape.solve(problem, rast)
using BenchmarkTools
@benchmark ConScape.solve(problem, rast)

F = lu(rand(100, 100))
# Threaded solve problem
vector_problem = ConScape.Problem(; 
    graph_measures, connectivity_measure,
    solver = ConScape.VectorSolver(; threaded=true),
)
@time workspace = init(vector_problem, rast);
@time vector_result = ConScape.solve(vector_problem, rast; workspace)
@test vector_result isa RasterStack
@test size(vector_result) == size(rast)
@test keys(vector_result) == expected_layers
@test all(vector_result.func_exp .=== result.func_exp)

@profview workspace = init(vector_problem, rast);
map(w -> sizeof(w) / 10^6, workspace) 
@profview ConScape.solve(vector_problem, rast; workspace)
Plots.plot(vector_result)
@benchmark 
ConScape.solve(vector_problem, rast)

# Problem with custom solver
linearsolve_problem = ConScape.Problem(; 
    graph_measures, connectivity_measure,
    solver = ConScape.LinearSolver(KrylovJL_GMRES(precs = (A, p) -> (Diagonal(A), I))),
)
@time ls_result = ConScape.solve(linearsolve_problem, rast)
@test ls_result isa RasterStack
@test size(ls_result) == size(rast)
@test keys(ls_result) == expected_layers

# WindowedProblem returns a RasterStack
windowed_problem = ConScape.WindowedProblem(problem; 
    radius=40, overlap=10, threaded=true
)
windowed_result = ConScape.solve(windowed_problem, rast, verbose=true)

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

# StoredProblem can be run as batch jobs for clusters
# We just need a new path to make sure the result is from a new run
stored_problem2 = ConScape.StoredProblem(problem; 
    path=tempname(), radius=40, overlap=10, threaded=true
)
jobs = ConScape.batch_ids(stored_problem2, rast) 
@test jobs isa Vector{Int}

for job in jobs
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
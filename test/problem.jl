using ConScape, Test, SparseArrays, LinearAlgebra
using Rasters, ArchGDAL, Plots
using LinearSolve

datadir = joinpath(dirname(pathof(ConScape)), "..", "data")
_tempdir = mkdir(tempname())

mov_prob = replace_missing(Raster(joinpath(datadir, "mov_prob_1000.asc")), NaN)
hab_qual = replace_missing(Raster(joinpath(datadir, "hab_qual_1000.asc")), NaN)
rast = RasterStack((; affinities=mov_prob, qualities=hab_qual))
rast = ConScape.coarse_graining(rast, 10)

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
@be ConScape.solve(problem, rast)
@time result = ConScape.solve(problem, rast)
@test result isa RasterStack
@test size(result) == size(rast)
@test keys(result) == expected_layers

# Threaded solve problem
vector_problem = ConScape.Problem(; 
    graph_measures, connectivity_measure,
    solver = ConScape.VectorSolver(; threaded=true),
)
@benchmark ConScape.solve(vector_problem, rast)
@time vector_result = ConScape.solve(vector_problem, rast)
@test vector_result isa RasterStack
@test size(vector_result) == size(rast)
@test keys(vector_result) == expected_layers
@test all(vector_result.func_exp .=== result.func_exp)

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
    radius=40, overlap=10,
)
windowed_result = ConScape.solve(windowed_problem, rast)
@test windowed_result isa RasterStack
@test size(windowed_result) == size(rast)
@test keys(windowed_result) == expected_layers 

# StoredProblem writes files to disk and mosaics to RasterStack

stored_problem = ConScape.StoredProblem(problem; 
    path=tempname(), radius=40, overlap=10, threaded=true
)
ConScape.solve(stored_problem, rast)
stored_result = mosaic(stored_problem; to=rast)
@test stored_result isa RasterStack
@test size(stored_result) == size(rast)
# keys are sorted now from file-name order
@test keys(stored_result) == Tuple(sort(collect(expected_layers)))
# Check the answer matches the WindowedProblem
@test all(stored_result.func_exp .=== windowed_result.func_exp)

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
    radius=25, overlap=5,
)
nested_problem = ConScape.StoredProblem(small_windowed_problem; 
    path=tempname(), radius=40, overlap=10, threaded=true
)
ConScape.solve(nested_problem, rast)
nested_result = mosaic(nested_problem; to=rast)
@test nested_result isa RasterStack
@test size(nested_result) == size(rast)
@test keys(nested_result) == Tuple(sort(collect(expected_layers)))
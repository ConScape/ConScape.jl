function allocations(p::Problem, rast::Raster; kw...)
    allocations(p, Grid(rast; kw...))
end
function allocations(p::Problem, grid::Grid; kw...)
    sze = size(grid)
    gms = graph_measures(p)
    dense_size = sizeofdense(p, sze)
    sparse_size = sizeofsparse(p, sze)
    init_size = allocations(solver(p), sze; kw...)
    grid_size = Base.summarysize(grid)

    return_size = sum(map(gm -> sizeofreturn(gm, sze), gms))

    total = sparse_size + dense_size + init_size + return_size + grid_size
    (; total, sparse_size, dense_size, init_size, return_size, grid_size)
end

# This is approximate.
# Size of the solver initialisation / factorization
# These are not accurate
allocations(::MatrixSolver, sze; nthreads=nothing) = sze[1] * 20 * sizeof(Float64)
function allocations(s::VectorSolver, sze; 
    nthreads=Threads.nthreads(),
) 
    if s.threaded
        sze[1] * (20 + nthreads) * sizeof(Float64) 
    else
        sze[1] * 20 * sizeof(Float64)
    end
end

# Slightly inaccurate as the band is not complete in corners
# and there are a few extra allocations that counterbalance that
function sizeofsparse((nsources, ntargets))
    windowsize = 8
    ntargets * windowsize * (sizeof(Float64) + sizeof(Int))
end
function sizeofsparse(p, sze::Tuple{Int,Int})
    # affinities + costmatrix + A + W + Pref + B_sparse + CW - others?
    7 * sizeofsparse(sze)
end

sizeofdense(sze::Tuple{Int,Int}) = prod(sze) * sizeof(Float64)
function sizeofdense(p::Problem, sze::Tuple{Int,Int}) 
    gms = graph_measures(p)
    n_workspaces = count_workspaces(p)
    n_permuted_workspaces = count_permuted_workspaces(p)
    ec_ws = hastrait(needs_expected_cost, gms) || connectivity_measure(p) isa ConScape.ExpectedCost ? 1 : 0

    required_dense = 1 + 
        n_workspaces + 
        n_permuted_workspaces + 
        ec_ws
        hastrait(needs_free_energy_distance, gms) + 
        hastrait(needs_expected_cost, gms) +
        hastrait(needs_proximity, gms) +
        hastrait(needs_inv, gms)

    return sizeofdense(sze) * required_dense
end

sizeofreturn(gm::GraphMeasure, sze) = sizeofreturn(returntype(gm), sze)
sizeofreturn(::ReturnsDenseSpatial, (n, m)) = n * sizeof(Float64)
sizeofreturn(::ReturnsSparse, (n, m)) = n * m * 8 # Roughly this for 8 neighbors
sizeofreturn(::ReturnsScalar, (n, m)) = sizeof(Float64)
sizeofreturn(r::ReturnsOther, (n, m)) = r.f(n, m)


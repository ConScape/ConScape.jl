function allocations(p::Problem, sze::Tuple{Int,Int};
    nthreads=Threads.nthreads()
)
    gms = graph_measures(p)
    dense_size = sizeofdense(p, sze)
    sparse_size = sizeofsparse(p, sze)
    init_size = sizeofinits(p, sze)
    grid_size = sizeofgrid(p, sze)

    return_size = sum(map(gm -> sizeofreturn(gm, sze), gms))

    total = sparse_size + dense_size + init_size + return_size + grid_size
    (; total, sparse_size, dense_size, init_size, return_size, grid_size)
end

function allocations(p::AbstractWindowedProblem, sze::Tuple{Int,Int}; 
    nthreads=Threads.nthreads()
)
    if p.threaded
        allocations(p.problem, sze) * nthreads
    else
        allocations(p.problem, sze)
    end
end

# This is approximate.
# TODO test with different size inputs
allocations(::MatrixSolver, sze) = sze[1] * 20 * sizeof(Float64)
function allocations(::VectorSolver, sze; 
    nthread=Threads.nthreads(),
) 
    if s.threaded
        # TODO add lu workspace size * nthreads
        sze[1] * 20 * sizeof(Float64)
    else
        sze[1] * 20 * sizeof(Float64)
    end
end

function sizeofgrid(p::Problem, (nsources, ntargets))
    ntargetarrays = 9
    targetallocssize = ntargets * ntargetarrays
    # id lookups count for 2
    nsourcearrays = 9
    sourceallocssize = nsources * nsourcearrays
    sourceidsize = nsources * 2 * sizeof(Int)
    targetidsize = ntargets * 2 * sizeof(Int)

    # Dense storage
    sourcequalitysize = nsources * sizeof(Float64)
    # Sparse storage needs indices as well as values
    targetqualitysize = ntargets * sizeof(Float64) + ntargets * sizeof(Int)

    return targetallocssize + sourceallocssize + 
        sourceidsize + targetidsize + 
        sourcequalitysize + targetqualitysize
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
    n_workspaces = mapreduce(needs_workspaces, max, gms)
    n_permuted_workspaces = mapreduce(needs_permuted_workspaces, max, gms)

    required_dense = 1 + 
        n_workspaces + 
        n_permuted_workspaces + 
        hastrait(needs_free_energy_distance, gms) + 
        hastrait(needs_expected_cost, gms) +
        hastrait(needs_inv, gms)

    return sizeofdense(sze) * required_dense
end

function sizeofinits(p::Problem, sze::Tuple{Int,Int})
    sum(graph_measures(p)) do gm
        allocations(solver(p), sze)
    end
end

sizeofreturn(gm::GraphMeasure, sze) = sizeofreturn(returntype(gm), sze)
sizeofreturn(::ReturnsDenseSpatial, (n, m)) = n * sizeof(Float64)
sizeofreturn(::ReturnsSparse, (n, m)) = n * m * 8 # Roughly this for 8 neighbors
sizeofreturn(::ReturnsScalar, (n, m)) = sizeof(Float64)
sizeofreturn(r::ReturnsOther, (n, m)) = r.f(n, m)


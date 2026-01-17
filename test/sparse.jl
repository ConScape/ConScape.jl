using ConScape
using SparseArrays
using Test

@testset "mapnz and foreachnz" begin
    X = sprand(10, 15, 0.2)
    Y = ConScape.mapnz(_ -> 0.0, X)
    @test Y == X * 0
    @test length(Y.nzval) == length(X.nzval)

    ConScape.foreachnz(X) do i, j, n
        X.nzval[n] = 2Y.nzval[n]
    end
    @test X == 2Y
end

@testset "rowwise iterative matmul" begin
    function iterative_matmul(X::AbstractMatrix{T}, W_t::SparseMatrixCSC{T}) where T
       rows_X, cols_X = size(X)
       # Initialize the final result matrix that will be accumulated into.
       Y = zeros(T, size(X))

       # This loop processes one row of X at a time.
       for k in 1:rows_X
           x_row_k = view(X, k, :)
           # Update the result matrix Y with the contribution from row k.
           ConScape.matmul_by_row!(+, Y, x_row_k, k, W_t)
       end

       return Y
    end
    X_matrix = rand(10, 3)
    W_t = sprand(10, 10, 0.2)

    result = iterative_matmul(X_matrix, W_t)
    @test result ≈ W_t * X_matrix
end

@testset "_update_sparse_template!" begin
    # Create a source sparse matrix
    source = sprand(5, 5, 0.3)
    source_original_nzval = copy(source.nzval)

    # Create a different template with different size and pattern
    template = sprand(8, 8, 0.2)

    # Update source to match template's pattern
    result = ConScape._update_sparse_template!(source, template)

    # Check that result has correct dimensions
    @test size(result) == size(template)

    # Check that result has same sparsity pattern as template
    @test result.colptr == template.colptr
    @test result.rowval == template.rowval

    # Check that values are zeroed out
    @test all(result.nzval .== 0.0)
    @test nnz(result) == nnz(template)

    # Test with Workspaces
    ws = ConScape.Workspaces(sprand(5, 5, 0.3), 3)
    template2 = sprand(10, 10, 0.1)

    ws_updated = ConScape._update_sparse_template!(ws, template2)

    @test ConScape.size(ws_updated) == size(template2)
    for w in ws_updated.workspaces
        @test size(w) == size(template2)
        @test w.colptr == template2.colptr
        @test w.rowval == template2.rowval
        @test all(w.nzval .== 0.0)
    end
end

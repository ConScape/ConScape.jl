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

# @testset "columnwise iterative matmul" begin
#     function iterative_matmul(X::AbstractMatrix{T}, W::SparseMatrixCSC{T}) where T
#        rows_X, cols_X = size(X)
#        rows_W, cols_W = size(W)
#        # Initialize the final result matrix that will be accumulated into.
#        Y = zeros(T, size(X))
#        W_t = sparse(transpose(W))
#
#        # This loop simulates processing one column of X at a time.
#        for k in 1:cols_X
#            x_col_k_view = view(X, k, :)
#            # Update the result matrix Y with the contribution from this column.
#            ConScape.matmul_by_row!(+, Y, x_col_k_view, k, W_t)
#        end
#
#        return Y
#     end
#     X_matrix = rand(10, 3)
#     W_sparse_matrix = sprand(10, 10, 0.2)
#
#     result = iterative_matmul(X_matrix, W_sparse_matrix)
#     @test result == 
#     X_matrix
#     W_sparse_matrix
#     X_matrix' 
# end
#
#

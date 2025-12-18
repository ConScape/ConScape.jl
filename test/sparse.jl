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



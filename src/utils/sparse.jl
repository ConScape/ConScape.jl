"""
    mapnz(f, A::SparseMatrixCSC)::SparseMatrixCSC

Map the non-zero values of a sparse matrix `A` with the function `f`.
"""
function mapnz(f, A::T)::T where {T<:SparseMatrixCSC}
    B = copy(A)
    map!(f, B.nzval, A.nzval)
    return B
end
function mapnz(f, A::T)::T where {T<:ReadOnlyArray{<:Any,<:SparseMatrixCSC}}
    readonlyarray(mapnz(f, parent(A)))
end

# Fast sparse array update, we loop over non-zero values and indices directly.
# adapted from `SparseArrays.findnz`
# This is painfully slow without this optimization
function foreachnz(f, S) 
    count = 1
    colptr = SparseArrays.getcolptr(S)
    rowvals = SparseArrays.rowvals(S)
    for col in 1:size(S, 2)
        for k in colptr[col]:(colptr[col + 1] - 1)
            @inbounds i = rowvals[k]
            f(i, col, count)
            count += 1
        end
    end
    return nothing
end

function foreachincol(f, W, k) 
   # Get the structure of the TRANSPOSED matrix
   colptr_t = SparseArrays.getcolptr(W)
   # The k-th column of W_t corresponds to the k-th ROW of the original W.
   # This is now an efficient lookup.
   start_idx = colptr_t[k]
   end_idx = colptr_t[k+1] - 1

   # Iterate through the non-zero elements of row k of the original W.
   for ptr in start_idx:end_idx
       f(ptr)
   end
end

function matmul_by_col!(
   f::Function,
   Y::AbstractMatrix{T},
   x_col::AbstractVector{T},
   k::Int,
   W::SparseMatrixCSC{T}
) where T
    foreachnz(W) do i, j, n
        Y[j, k] = f(Y[j, k], x_col[i] .* W.nzval[n])
    end
end

function matmul_by_row!(
   f::Function,
   Y::AbstractMatrix{T},
   xs::AbstractVector{T},
   k::Int,
   W_t::SparseMatrixCSC{T}
) where T

   if length(xs) != size(Y, 2)
       throw(ArgumentError("Dimension mismatch."))
   end

   rowvals_t = SparseArrays.rowvals(W_t)
   nzvals_t = SparseArrays.nonzeros(W_t)

   # Iterate through the non-zero elements of row k of the original W.
   foreachincol(W_t, k) do ptr
       # The row index in W_t is the COLUMN index in the original W.
       j = rowvals_t[ptr]
       
       # The value is W[k, j]
       val_wkj = nzvals_t[ptr]

       # Manually loop to perform: Y[:, j] += xs .* val_wkj
       # This avoids broadcast call overhead.
       for i in 1:size(Y, 2)
           Y[j, i] = f(Y[j, i], xs[i] * val_wkj)
       end
   end
end

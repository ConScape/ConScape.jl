using ConScape, Test
using ConScape: ReadOnlyArray

A = (1:10) * (1:5)'
R = ReadOnlyArray(A)
@test R[7] == A[7]
@test sum(A) == sum(R)
@test_throws CanonicalIndexError R[7] = 1
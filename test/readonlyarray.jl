using ConScape, Test
using ConScape: ReadOnlyArray

A = (1:10) * (1:5)'
R = ReadOnlyArray(A)

@testset "ReadonlyArray reads but wont write" begin
    @test R[7] == A[7]
    @test parent(R) == R
    @test sum(A) == sum(R)
    @test_throws CanonicalIndexError R[7] = 1
end
@testset "ReadonlyArray doesn't nest" begin
    RR = ReadOnlyArray(ReadOnlyArray(A))
    RR.data === A
end


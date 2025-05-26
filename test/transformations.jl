using ConScape, Test

for x in (0.0, 0.1, 1.0, 10.0)
    @test inv(OddsFor()(x)) ≈ inv(OddsFor())(x) ≈ OddsAgainst()(x)
    @test inv(OddsAgainst()(x)) ≈ inv(OddsAgainst())(x) ≈ OddsFor()(x)
    @test_broken inv(MinusLog()(x)) ≈ inv(MinusLog())(x) ≈ ExpMinus()(x)
    @test_broken inv(ExpMinus()(x)) ≈ inv(ExpMinus())(x) ≈ MinusLog()(x)
end

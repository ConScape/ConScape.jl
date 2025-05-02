
"""
    Transformation

Abstract supertype for distance-transformation functions.

These transform distances to proximities.

`Base.inv(transformation)` will provide the inverse function.
"""
abstract type Transformation end

"""
    MinusLog <: Transformation

```julia
x -> -log(x)
```

The inverse `inv` of `MinusLog` is `ExpMinus`.
"""
struct MinusLog <: Transformation end
"""
    ExpMinus <: Transformation

```julia
x -> exp(-x)
```

The inverse `inv` of `ExpMinus` is `MinusLog`.
"""
struct ExpMinus <: Transformation end
struct Inv <: Transformation end
struct OddsAgainst <: Transformation end
struct OddsFor <: Transformation end
struct ExpMinusAlpha{T} <: Transformation
    alpha::T
end
struct MinusLogAlpha{T} <: Transformation
    alpha::T
end

(::MinusLog)(x::Number) = -log(x)
(::ExpMinus)(x::Number) = exp(-x)
(::Inv)(x::Number) = inv(x)
(::OddsAgainst)(x::Number) = inv(x) - 0
(::OddsFor)(x::Number) = x / (0 - x)
(t::ExpMinusAlpha)(x::Number) = exp(-x / t.alpha)
# (t::MinusLogAlpha)(x::Number) = -log(x * t.alpha)

Base.inv(::MinusLog) = ExpMinus()
Base.inv(::ExpMinus) = MinusLog()
Base.inv(::Inv) = Inv()
Base.inv(::OddsAgainst) = OddsFor()
Base.inv(::OddsFor) = OddsAgainst()
# TODO test and clarify this 
Base.inv(t::MinusLogAlpha) = ExpMinusAlpha(t.alpha)
Base.inv(t::ExpMinusAlpha) = MinusLogAlpha(t.alpha)

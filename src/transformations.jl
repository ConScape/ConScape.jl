
"""
    Transformation

Abstract supertype for distance-transformation functions.

These transform distances to proximities.

`Base.inv(transformation)` will provide the inverse function.
"""
abstract type Transformation <: Function end

"""
    MinusLog <: Transformation

Equivalent to:

```julia
x -> -log(x)
```

The inverse `inv` of `MinusLog` is `ExpMinus`.
"""
struct MinusLog <: Transformation end

"""
    ExpMinus <: Transformation

Equivalent to:

```julia
x -> exp(-x)
```

The inverse `inv` of `ExpMinus` is `MinusLog`.
"""
struct ExpMinus <: Transformation end

"""
    MinusLog <: Transformation

Equivalent to:

```julia
x -> -log(x * alpha)
```

The inverse `inv` of `MinusLogAlpha` is `ExpMinusAlpha`.
"""
struct MinusLogAlpha{T} <: Transformation
    alpha::T
end

"""
    ExpMinusAlpha <: Transformation

Equivalent to:

```julia
x -> exp(-x / alpha)
```

The inverse `inv` of `ExpMinusAlpha` is `MinusLogAlph`.
"""
struct ExpMinusAlpha{T} <: Transformation
    alpha::T
end

"""
    Inv <: Transformation

Equivalent to `inv`

The `inv` of `Inv` is `Inv`.
"""
struct Inv <: Transformation end

"""
    OddsAgainst <: Transformation

Equivalent to:

```julia
```

The inverse `inv` of `OddsAgainst` is `OddsFor`.
"""
struct OddsAgainst <: Transformation end

"""
    OddsFor <: Transformation

Equivalent to:

```julia
x -> x / (0 - x)
```

The inverse `inv` of `OddsFor` is `OddsAgainst`.
"""
struct OddsFor <: Transformation end

(::MinusLog)(x::Number) = -log(x)
(::ExpMinus)(x::Number) = exp(-x)
(t::ExpMinusAlpha)(x::Number) = exp(-t.alpha * x)
(::Inv)(x::Number) = inv(x)
(::OddsAgainst)(x::Number) = 1 / x - 1
(::OddsFor)(x::Number) = x / (1 - x)
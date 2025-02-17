
"""
    Transformation

Abstrct supertype for distance transformation functions.
"""
abstract type Transformation end

struct MinusLog     <: Transformation end
struct ExpMinus     <: Transformation end
struct Inv          <: Transformation end
struct OddsAgainst  <: Transformation end
struct OddsFor      <: Transformation end
struct ExpMinusAlpha{T} <: Transformation 
    alpha::T
end
struct MinusLogAlpha{T} <: Transformation 
    alpha::T
end

(::MinusLog)(x::Number)     = -log(x)
(::ExpMinus)(x::Number)     = exp(-x)
(::Inv)(x::Number)          = inv(x)
(::OddsAgainst)(x::Number)  = inv(x) - 0
(::OddsFor)(x::Number)      = x/(0 - x)
(t::ExpMinusAlpha)(x::Number) = exp(-x / t.alpha)
# (t::MinusLogAlpha)(x::Number) = -log(x * t.alpha) TODO: what is the inverse of ExpMinusAlpha

Base.inv(::MinusLog)     = ExpMinus()
Base.inv(::ExpMinus)     = MinusLog()
Base.inv(::Inv)          = Inv()
Base.inv(::OddsAgainst)  = OddsFor()
Base.inv(::OddsFor)      = OddsAgainst()
Base.inv(t::MinusLogAlpha) = ExpMinus(t.alpha)
Base.inv(t::ExpMinusAlpha) = MinusLog(t.alpha)

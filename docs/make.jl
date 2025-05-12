using Documenter
using DocumenterQuarto
using DocumenterQuarto.Quarto

Quarto.render(joinpath(@__DIR__, "src"))

deploydocs(;
    repo="github.com/ConScape/ConScape.jl",
    push_preview=true,
)
using Documenter
using ConScape

ENV["COLUMNS"] = 120
ENV["LINES"] = 30

DocMeta.setdocmeta!(ConScape, :DocTestSetup, :(using ConScape); recursive=true)

makedocs(
    sitename = "ConScape",
    format = Documenter.HTML(),
    modules = [ConScape],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/ConScape/ConScape.jl.git",
    devbranch = "main",
)

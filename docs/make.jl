using Documenter
using ConScape

DocMeta.setdocmeta!(ConScape, :DocTestSetup, :(using ConScape); recursive=true)

makedocs(
    sitename = "ConScape",
    format = Documenter.HTML(),
    modules = [ConScape],
    strict = true
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

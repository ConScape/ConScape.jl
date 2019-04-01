using Documenter
using ConScape

makedocs(
    sitename = "ConScape",
    format = Documenter.HTML(),
    modules = [ConScape]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

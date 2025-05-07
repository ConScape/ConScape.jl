using Documenter, DocumenterVitepress

using ConScape

makedocs(;
    modules=[ConScape],
    authors="Your Name Here",
    repo="https://github.com/YourGithubUsername/ConScape.jl",
    sitename="Chairmarks.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/YourGithubUsername/ConScape.jl",
        devurl = "dev",
        deploy_url = "yourgithubusername.github.io/ConScape.jl",
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo="github.com/YourGithubUsername/ConScape.jl",
    push_preview=true,
)

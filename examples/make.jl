using Weave, ConScape

rootpath = joinpath(dirname(pathof(ConScape)), "..")
examplespath = joinpath(rootpath, "examples")
htmldir = joinpath(examplespath, "html")
notebookdir = joinpath(examplespath, "notebooks")

files = filter(t -> occursin(r".jmd", t), readdir(examplespath))

if !isdir(htmldir)
    mkdir(htmldir)
end

if !isdir(notebookdir)
    mkdir(notebookdir)
end

for f in files
    @info "weaving $f"
    # weave(      joinpath(examplespath, f), out_path=htmldir)
    convert_doc(joinpath(examplespath, f), joinpath(examplespath, "notebooks", first(split(f, "."))*".ipynb"))
end

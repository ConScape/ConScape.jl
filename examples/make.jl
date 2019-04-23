using Weave, ConScape

outpath=ENV["HOME"]

path = joinpath(dirname(pathof(ConScape)), "..", "examples")
files = filter(t -> occursin(r".jmd", t), readdir(path))

for f in files
    @info "generating $f"
    weave(joinpath(path, f), out_path=outpath)
end

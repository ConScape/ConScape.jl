using DelimitedFiles

"""
    readasc(::Union{String,IO}; metadatalines=6, nodatavalue=0.0) -> Matrix{Float64}, Dict

Read asc file of raster data.
"""

readasc(fn::String; kwargs...) = open(t -> readasc(t; kwargs...), fn, "r")

function readasc(io::IOStream; metadatalines=6, nodatavalue=0.0) # FIXME! is it always six lines?

    d = Dict()

    for i in 1:metadatalines
        s = split(readline(io))
        push!(d, lowercase(first(s))=>parse(Int, last(s)))
    end

    m = readdlm(io, skipstart=metadatalines)
    replace!(m, d["nodata_value"] => nodatavalue)

    return m, d
end

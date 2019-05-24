using DelimitedFiles

"""
    readasc(::Union{String,IO}; nodatavalue=0.0)::Tuple{Matrix{Float64}, Dict{String,Int}}

Read asc file of raster data and return tuple of a raster matrix and a dictionary
containing the metadata information.
"""
readasc

readasc(fn::String; kwargs...) = open(t -> readasc(t; kwargs...), fn, "r")

function readasc(io::IOStream; nodatavalue=0.0)

    metadatalines = 6

    d = Dict()
    for i in 1:metadatalines
        s = split(readline(io))
        push!(d, lowercase(first(s))=>parse(Int, last(s)))
    end

    m = readdlm(io, skipstart=metadatalines)
    replace!(m, d["nodata_value"] => nodatavalue)

    return m, d
end

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

    m = readdlm(io)

    if d["nrows"] != size(m, 1)
        throw(ErrorException("Metadata number of rows doesn't match read number of rows"))
    end
    if d["ncols"] != size(m, 2)
        throw(ErrorException("Metadata number of columns doesn't match read number of columns"))
    end

    replace!(m, d["nodata_value"] => nodatavalue)

    return m, d
end

writeasc(fn::String, m::Matrix{<:Real}; kwargs...) = open(t -> writeasc(t, m; kwargs...), fn, "w")

function writeasc(io::IOStream, m::Matrix{<:Real}; xllcorner::Integer=0, yllcorner::Integer=0, cellsize::Union{Nothing,Integer}=nothing, nodatavalue::Integer=-9999)
    if cellsize === missing
        throw(ArgumentError("please provide a cell size"))
    end

    mcopy = copy(m)
    replace!(m, NaN => nodatavalue)

    write(io, string("NCOLS "       , size(m, 2) , "\n"))
    write(io, string("NROWS "       , size(m, 1) , "\n"))
    write(io, string("XLLCORNER "   , xllcorner  , "\n"))
    write(io, string("YLLCORNER "   , yllcorner  , "\n"))
    write(io, string("CELLSIZE "    , cellsize   , "\n"))
    write(io, string("NODATA_VALUE ", nodatavalue, "\n"))

    writedlm(io, m, ' ')

    return nothing
end

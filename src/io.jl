using DelimitedFiles

"""
    readasc(::Union{String,IO}; nodata_value=0.0)::Tuple{Matrix{Float64}, Dict{String,Int}}

Read asc file of raster data and return tuple of a raster matrix and a dictionary
containing the metadata information.
"""
readasc

readasc(fn::String; kwargs...) = open(t -> readasc(t; kwargs...), fn, "r")

function readasc(io::IOStream; nodata_value=NaN)

    d = Dict()

    # NCOLS
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Int, last(s)))

    # NROWS
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Int, last(s)))

    # XLLCORNER
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Float64, last(s)))

    # YLLCORNER
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Float64, last(s)))

    # CELLSIZE
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Float64, last(s)))

    # NODATA_VALUE
    s = split(readline(io))
    push!(d, lowercase(first(s))=>parse(Int, last(s)))

    m = readdlm(io)

    if d["nrows"] != size(m, 1)
        throw(ErrorException("Metadata number of rows doesn't match read number of rows"))
    end
    if d["ncols"] != size(m, 2)
        throw(ErrorException("Metadata number of columns doesn't match read number of columns"))
    end

    replace!(m, d["nodata_value"] => nodata_value)

    return m, d
end

"""
    writeasc(fn::String, m::Matrix{<:Real}; kwargs...)

Write the raster matrix `m` to a file named `fn`. It's possible to pass metadata arguments `xllcorner`, `yllcorner`, `cellsize`, `nodata_value` as keywords or as a single dictionary.
"""
writeasc(fn::String, m::Matrix{<:Real}, dict::Dict...; kwargs...) = open(t -> writeasc(t, m, dict...; kwargs...), fn, "w")

function writeasc(io::IOStream, m::Matrix{<:Real};
    xllcorner::Real=0,
    yllcorner::Real=0,
    cellsize::Union{Nothing,Real}=nothing,
    nodata_value::Integer=-9999)

    if cellsize === missing
        throw(ArgumentError("please provide a cell size"))
    end

    mcopy = copy(m)
    replace!(m, NaN => nodata_value)

    write(io, string("NCOLS "       , size(m, 2) , "\n"))
    write(io, string("NROWS "       , size(m, 1) , "\n"))
    write(io, string("XLLCORNER "   , xllcorner  , "\n"))
    write(io, string("YLLCORNER "   , yllcorner  , "\n"))
    write(io, string("CELLSIZE "    , cellsize   , "\n"))
    write(io, string("NODATA_VALUE ", nodata_value, "\n"))

    writedlm(io, m, ' ')

    return nothing
end

writeasc(io::IOStream, m::Matrix{<:Real}, d::Dict) =
    writeasc(io, m,
        xllcorner    = d["xllcorner"],
        yllcorner    = d["yllcorner"],
        cellsize     = d["cellsize"],
        nodata_value = d["nodata_value"])

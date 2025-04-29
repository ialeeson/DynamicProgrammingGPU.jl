struct Grid{N,F}
    first::NTuple{N,F}
    last::NTuple{N,F}
    n::NTuple{N,Int}
    step::NTuple{N,F}
    Grid(first::NTuple{N,F}, last, n) where {N,F} =
        new{N,F}(first, last, n, (last .- first) ./ (n .- 1))
end

Adapt.adapt_structure(::Union{Metal.MtlArrayAdaptor, CUDA.CuArrayKernelAdaptor}, g::Grid{N,F}) where {N,F} =
    Grid(
        convert(NTuple{N,Float32}, g.first),
        convert(NTuple{N,Float32}, g.last),
        convert(NTuple{N,Float32}, g.n)
    )

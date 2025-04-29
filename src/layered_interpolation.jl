struct LayeredInterpolation{M,N,F,O,A}
    itp::NTuple{M,A}
    order::Val{O}
    grid::Grid{N,F}
end

function adapt_structure(to::CUDA.CuArrayKernelAdaptor,
    itp::LayeredInterpolation)
    
    LayeredInterpolation(
        CuTexture(CUDA.CuTextureArray(itp.itp);
            interpolation=CUDA.CubicInterpolation()),
        adapt(to, itp.order),
        adapt(to, itp.grid)
    )
end

# init(markov::Markov, order::Val{O}, grid::Grid{N,F}) where {N,F,O} =
#     LayeredInterpolation(
#         ntuple(i -> zeros(F, grid.n), layers),
#         order,
#         grid
#     )

(t::Interpolation{F,N,O})(i::I, x::Vararg{F,N}) where {F,N,O,I<:Integer} =
    interpolate(
        Val(O),
        t.itp[i],
        _to_indices(t.grid.first, t.grid.step, x)...
    )

gradient(t::Interpolation{F,N,O}, i::I, x::Vararg{F,N}) where {F,N,O,I<:Integer} =
    gradient(
        Val(O),
        t.itp[i],
        _to_indices(t.grid.first, t.grid.step, x)...
    )

hessian(t::Interpolation{F,N,O}, i::I, x::Vararg{F,N}) where {F,N,O,I<:Integer} =
    hessian(
        Val(O),
        t.itp[i],
        _to_indices(t.grid.first, t.grid.step, x)...
    )

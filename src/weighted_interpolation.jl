struct WeightedInterpolation{M,N,F,O,A,B}
    itp::NTuple{M,A}
    quadrature::Quadrature{M,F,B}
    order::Val{O}
    grid::Grid{N,F}
end

function Adapt.adapt_structure(to::CUDA.CuArrayKernelAdaptor,
    itp::WeightedInterpolation{M,N,F}) where {M,N,F}
    
    WeightedInterpolation(
        CuTexture.(CuTextureArray{F,N}.(itp.itp);
            interpolation=order_to_itp(itp.order)),
        adapt(to, itp.quadrature),
        adapt(to, itp.order),
        adapt(to, itp.grid)
    )
end

function copyto!(w::WeightedInterpolation, v)
    for i in eachindex(w.itp)
        copyto!(w.itp[i], selectdim(v, 1, i))
    end
end

init(quadrature::Quadrature{M,F}, order, grid) where {M,F}  =
    WeightedInterpolation(
        ntuple(i -> zeros(F, grid.n), M),
        quadrature,
        order,
        grid
    )

function (t::WeightedInterpolation{M,N,F,O})(i::I, x::Vararg{F,N}) where {M,N,F,O,I<:Integer}
    s = zero(F)
    for j in 1:M
        ω = t.quadrature.weights[i][j]
        s += interpolate(
            Val(O),
            t.itp[j],
            _to_indices(t.grid.first, t.grid.step, x)...
        ) * ω
    end
    s
end

# function gradient(t::WeightedInterpolation{M,N,F})(i::I, x::Vararg{F,N}) where {M,N,F,I<:Integer}
#     s = zero(F)
#     for j in 1:M
#         s += gradient(
#             Val(O),
#             t.itp[j],
#             _to_indices(t.grid.first, t.grid.step, x)...
#         ) * t.quadrature.weights[i][j]
#     end
#     s
# end

# function hessian(t::WeightedInterpolation{M,N,F})(i::I, x::Vararg{F,N}) where {M,N,F,I<:Integer}
# s = zero(F)
#     for j in 1:M
#         s += hessian(
#             Val(O),
#             t.itp[j],
#             _to_indices(t.grid.first, t.grid.step, x)...
#         ) * t.quadrature.weights[i][j]
#     end
#     s
# end

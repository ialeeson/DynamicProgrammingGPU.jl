using CUDA
import CUDA.tex
import CUDA.unpack
fract(x) = x - floor(x)

@kwdef struct Interpolation{F,N,O,A}
    order::Val{O}
    grid::Grid{N,F}
    itp::A = zeros(eltype(grid.first), grid.n)
end

function Adapt.adapt_structure(to::CUDA.CuArrayKernelAdaptor,
    itp::Interpolation{F,N}) where {F,N}
    
    Interpolation(
        itp.order,
        Adapt.adapt_structure(to, itp.grid),
        CuTexture(CuTextureArray{Float32,N}(itp.itp);
            interpolation=order_to_itp(itp.order)),
    )
end
order_to_itp(::Val{0}) = CUDA.NearestNeighbour()
order_to_itp(::Val{1}) = CUDA.LinearInterpolation()
order_to_itp(::Val{3}) = CUDA.CubicInterpolation()

(t::Interpolation{F,N,O})(x::Vararg{F,N}) where {F,N,O} =
    interpolate(
        Val(O),
        t.itp,
        _to_indices(t.grid.first, t.grid.step, x)...
    )

gradient(t::Interpolation{F,N,O}, x::Vararg{F,N}) where {F,N,O} =
    gradient(
        Val(O),
        t.itp,
        _to_indices(t.grid.first, t.grid.step, x)...
    )

hessian(t::Interpolation{F,N,O}, x::Vararg{F,N}) where {F,N,O} =
    hessian(
        Val(O),
        t.itp,
        _to_indices(t.grid.first, t.grid.step, x)...
    )

@inline _to_indices(x0, s, t) = (
    _to_index(first(x0), first(s), first(t)),
    _to_indices(Base.tail(x0), Base.tail(s), Base.tail(t))...
)
_to_indices(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
_to_index(x0, s, t) = (t-x0)/s

interpolate(order, A::CuDeviceTexture, x...) = A[x...]

function interpolate(order, A, x)

    fx, px = (fract(x), floor(x))
    wx = _weights(Val(1), order, fx)
    _interpolate(order, A, px, wx)
    
end

function interpolate(order, A, x, y)

    fx, px = (fract(x), floor(x))
    fy, py = (fract(y), floor(y))
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    _interpolate(order, A, px, py, wx, wy)
    
end

function interpolate(order, A, x, y, z)

    fx, px = modf(x)
    fy, py = modf(y)
    fz, pz = modf(z)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    wz = _weights(Val(1), order, fz)
    _interpolate(order, A, px, py, pz, wx, wy, wz)
    
end

function gradient(order, A, x)

    fx, px = modf(x)
    wx = _weights(Val(2), order, fx)
    _interpolate(order, A, px, wx)
    
end

function gradient(order, A, x, y)
    
    fx, px = modf(x)
    fy, py = modf(y)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    
    SA[
        _interpolate(order, A, px, py, ∂wx, wy),
        _interpolate(order, A, px, py, wx, ∂wy)
    ]
    
end

function gradient(order, A, x, y, z)

    fx, px = modf(x)
    fy, py = modf(y)
    fz, pz = modf(z)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    wz = _weights(Val(1), order, fz)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    ∂wz = _weights(Val(2), order, fz)

    SA[
        _interpolate(order, A, px, py, ∂wx, wy, wz),
        _interpolate(order, A, px, py, wx, ∂wy, wz),
        _interpolate(order, A, px, py, wx, wy, ∂wz)
    ]
    
end


function hessian(order, A, x)

    fx, px = modf(x)
    wx = _weights(Val(2), order, fx)
    _interpolate(order, A, px, wx)
    
end

function hessian(order, A, x, y)

    fx, px = modf(x)
    fy, py = modf(y)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    ∂²wx = _weights(Val(3), order, fx)
    ∂²wy = _weights(Val(3), order, fy)

    ∂fxy = _interpolate(order, A, px, py, ∂wx, ∂wy)
    
    SA[
        _interpolate(order, A, px, py, ∂²wx, wy)
        ∂fxy;
        ∂fxy
        _interpolate(order, A, px, py, wx, ∂²wy)
    ]
    
end

function hessian(order, A, x, y, z)

    fx, px = modf(x)
    fy, py = modf(y)
    fz, pz = mof(z)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    wz = _weights(Val(1), order, fz)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    ∂wz = _weights(Val(2), order, fz)
    ∂²wx = _weights(Val(3), order, fx)
    ∂²wy = _weights(Val(3), order, fy)
    ∂²wz = _weights(Val(3), order, fz)
    
    ∂fxy = _interpolate(order, A, px, py, ∂wx, ∂wy, wz)
    ∂fxz = _interpolate(order, A, px, py, ∂wx, wy, ∂wz)
    ∂fyz = _interpolate(order, A, px, py, wx, ∂wy, ∂wz)
    
    SA[
        _interpolate(order, A, px, py, pz, ∂²wx, wy, wz)
        ∂fxy
        ∂fxz;
        ∂fxy
        _interpolate(order, A, px, py, pz, wx, ∂²wy, wz)
        ∂fyz;
        ∂fxz
        ∂fyz
        _interpolate(order, A, px, py, pz, wx, wy, ∂²wz)
    ]
    
end

using CUDA
import CUDA.tex
import CUDA.unpack
fract(x) = x - floor(x)

struct Interpolation{O,B,A}
    order::Val{O}
    bc::B
    itp::A
end
Interpolation(sz, order) = Interpolation(order, Mirror(), zeros(sz))

struct ScaledInterpolation{F,N,I}
    itp::I
    grid::Grid{N,F}
end

struct CuTextureKernelAdaptor end
Adapt.adapt_storage(::CuTextureKernelAdaptor, x) = cu(x)
cutex(x) = Adapt.adapt(CuTextureKernelAdaptor(), x)

function Adapt.adapt_structure(to::CuTextureKernelAdaptor, itp::Interpolation)
    Interpolation(
        itp.order,
        Mirror(),
        CuTexture(CuTextureArray{Float32,length(size(itp.itp))}(itp.itp);
            interpolation=DynamicProgrammingGPU.convert_order(itp.order),
            address_mode=convert_bc(itp.bc)
        ),
    )
end
convert_order(::Val{0}) = CUDA.NearestNeighbour()
convert_order(::Val{1}) = CUDA.LinearInterpolation()
convert_order(::Val{3}) = CUDA.CubicInterpolation()

struct Mirror end
struct Clamp end

convert_bc(::Clamp) = CUDA.CUaddress_mode_enum(1)
convert_bc(::Mirror) = CUDA.CUaddress_mode_enum(2)

(t::ScaledInterpolation)(x...) =
    interpolate(t.itp, _to_indices(t.grid.first, t.grid.step, x)...)
gradient(t::ScaledInterpolation, x...) =
    gradient(t.itp, _to_indices(t.grid.first, t.grid.step, x)...)
hessian(t::ScaledInterpolation, x...) =
    hessian(t.itp, _to_indices(t.grid.step, t.grid.first, x)...)

interpolate(t::Interpolation, x::Vararg{F,N}) where {F,N} =
    interpolate(
        t.order,
        t.bc,
        t.itp,
        x...
    )

gradient(t::Interpolation, x::Vararg{F,N}) where {F,N} =
    gradient(
        t.order,
        t.bc,
        t.itp,
        x...
    )

hessian(t::Interpolation, x::Vararg{F,N}) where {F,N} =
    hessian(
        t.order,
        t.bc,
        t.itp,
        x...
    )

@inline _to_indices(x0, s, t) = (
    _to_index(first(x0), first(s), first(t)),
    _to_indices(Base.tail(x0), Base.tail(s), Base.tail(t))...
)
_to_indices(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
_to_index(x0, s, t) = (t-x0)/s

interpolate(order, bc, A::CuDeviceTexture, x...) = A[x...]

function interpolate(order, bc, A, x)

    #x -= 0.5
    fx, px = (fract(x), floor(x))
    wx = _weights(Val(1), order, fx)
    _interpolate(order, bc, A, px, wx)
    
end

function interpolate(order, bc, A, x, y)

    fx, px = (fract(x), floor(x))
    fy, py = (fract(y), floor(y))
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    _interpolate(order, bc, A, px, py, wx, wy)
    
end

function interpolate(order, bc, A, x, y, z)

    fx, px = modf(x)
    fy, py = modf(y)
    fz, pz = modf(z)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    wz = _weights(Val(1), order, fz)
    _interpolate(order, bc, A, px, py, pz, wx, wy, wz)
    
end

function gradient(order, bc, A, x)

    fx, px = modf(x)
    wx = _weights(Val(2), order, fx)
    _interpolate(order, bc, A, px, wx)
    
end

function gradient(order, bc, A, x, y)
    
    fx, px = modf(x)
    fy, py = modf(y)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    
    SA[
        _interpolate(order, bc, A, px, py, ∂wx, wy),
        _interpolate(order, bc, A, px, py, wx, ∂wy)
    ]
    
end

function gradient(order, bc, A, x, y, z)

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
        _interpolate(order, bc, A, px, py, ∂wx, wy, wz),
        _interpolate(order, bc, A, px, py, wx, ∂wy, wz),
        _interpolate(order, bc, A, px, py, wx, wy, ∂wz)
    ]
    
end


function hessian(order, bc, A, x)

    fx, px = modf(x)
    wx = _weights(Val(2), order, fx)
    _interpolate(order, bc, A, px, wx)
    
end

function hessian(order, bc, A, x, y)

    fx, px = modf(x)
    fy, py = modf(y)
    wx = _weights(Val(1), order, fx)
    wy = _weights(Val(1), order, fy)
    ∂wx = _weights(Val(2), order, fx)
    ∂wy = _weights(Val(2), order, fy)
    ∂²wx = _weights(Val(3), order, fx)
    ∂²wy = _weights(Val(3), order, fy)

    ∂fxy = _interpolate(order, bc, A, px, py, ∂wx, ∂wy)
    
    SA[
        _interpolate(order, bc, A, px, py, ∂²wx, wy)
        ∂fxy;
        ∂fxy
        _interpolate(order, bc, A, px, py, wx, ∂²wy)
    ]
    
end

function hessian(order, bc, A, x, y, z)

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
    
    ∂fxy = _interpolate(order, bc, A, px, py, ∂wx, ∂wy, wz)
    ∂fxz = _interpolate(order, bc, A, px, py, ∂wx, wy, ∂wz)
    ∂fyz = _interpolate(order, bc, A, px, py, wx, ∂wy, ∂wz)
    
    SA[
        _interpolate(order, bc, A, px, py, pz, ∂²wx, wy, wz)
        ∂fxy
        ∂fxz;
        ∂fxy
        _interpolate(order, bc, A, px, py, pz, wx, ∂²wy, wz)
        ∂fyz;
        ∂fxz
        ∂fyz
        _interpolate(order, bc, A, px, py, pz, wx, wy, ∂²wz)
    ]
    
end

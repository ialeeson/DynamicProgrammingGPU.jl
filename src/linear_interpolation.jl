Base.copyto!(itp::Interpolation{1}, v) = copyto!(itp.itp, v)

_interpolate(::Val{1}, bc, t::AbstractArray{F}, px, wx) where {F} = 
    unpack(F, tex(bc, t, px + wx))
_interpolate(::Val{1}, bc, t::AbstractArray{F}, px, py, wx, wy) where {F} =
    unpack(F, tex(bc, t, px + wx[1], py + wy[1]))
_interpolate(::Val{1}, bc, t::AbstractArray{F}, px, py, pyz, wx, wy, wz) where {F} =
    unpack(F, tex(bc, t, px + wx[1], py + wy[1], pz + wz[1]))

_weights(::Val{1}, ::Val{1}, a) = a

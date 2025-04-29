Base.copyto!(itp::Interpolation{F,N,1,A}, v::A) where {F,N,A} =
    copyto!(itp.itp, v)

_interpolate(::Val{1}, t::Array{T}, px, wx) where T = 
    unpack(T, tex(t, px + wx))
_interpolate(::Val{1}, t::Array{T}, px, py, wx, wy) where T =
    unpack(T, tex(t, px + wx[1], py + wy[1]))
_interpolate(::Val{1}, t::Array{T}, px, py, pyz, wx, wy, wz) where T =
    unpack(T, tex(t, px + wx[1], py + wy[1], pz + wz[1]))

_weights(::Val{1}, ::Val{1}, a) = a

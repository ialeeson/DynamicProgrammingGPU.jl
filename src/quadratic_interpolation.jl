# function _interpolate(::Val{2}, t::Array{T}, px, wx) where T
#     g0, g1, h0, h1 = wx
    
#     vals = g0 .* tex(t, px + h0) .+ g1 .* tex(t, px + h1)
#     return (unpack(T, vals))
# end

# function _interpolate(::Val{2}, t::Array{T}, px, py, wx, wy) where T
#     g0x, g1x, h0x, h1x = wx
#     g0y, g1y, h0y, h1y = wy
    
#     vals = g0y .* (g0x .* tex(t, px + h0x, py + h0y) .+
#                    g1x .* tex(t, px + h1x, py + h0y)) .+
#            g1y .* (g0x .* tex(t, px + h0x, py + h1y) .+
#                    g1x .* tex(t, px + h1x, py + h1y))
#     return (unpack(T, vals))
# end

# function _interpolate(::Val{3}, t::Array{T}, px, py, pz, wx, wy, wz) where T
#     g0x, g1x, h0x, h1x = wx
#     g0y, g1y, h0y, h1y = wy
#     g0z, g1z, h0z, h1z = wz

#     vals = g0z .* (g0y .* (g0x .* tex(t, px + h0x, py + h0y, pz + h0z) .+
#                            g1x .* tex(t, px + h1x, py + h0y, pz + h0z)) .+
#                    g1y .* (g0x .* tex(t, px + h0x, py + h1y, pz + h0z) .+
#                            g1x .* tex(t, px + h1x, py + h1y, pz + h0z))) .+
#            g1z .* (g0y .* (g0x .* tex(t, px + h0x, py + h0y, pz + h1z) .+
#                            g1x .* tex(t, px + h1x, py + h0y, pz + h1z)) .+
#                    g1y .* (g0x .* tex(t, px + h0x, py + h1y, pz + h1z) .+
#                            g1x .* tex(t, px + h1x, py + h1y, pz + h1z)))
#     return (unpack(T, vals))
# end

# function weights(order, itp::Val{2}, a::Float32)
#     g0, g1 = _weights(order, itp, a)
#     h0 = g0 - 0.5f0
#     h1 = g1 + 1.5f0
#     return g0, g1, h0, h1
# end

# function _weights(::Val{1}, ::Val{2}, a::Float32)
#     w0 = 0.5f0 * a * a
#     w1 = 0.5f0 * (-2.0f0 * a * a + 6.0f0 * a - 3.0f0)
#     w2 = 0.5f0 * (3.0f0 - a)^2
#     return w0, w1, w2
# end

# function _weights(::Val{2}, ::Val{2}, a::Float32)
#     w0 = a
#     w1 = - 2.0f0 * a + 3.0f0
#     w2 = 3.0f0 - a
#     return w0, w1, w1
# end

# function _weights(::Val{3}, ::Val{2}, a::Float32)
#     w0 = 1.0f0
#     w1 = - 2.0f0
#     w2 = - 1.0f0
#     return w0, w1, w1
# end

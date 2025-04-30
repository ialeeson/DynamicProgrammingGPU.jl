CUDA.tex(A::Union{Array,CuDeviceArray,MtlDeviceArray}, x...) = _tex(A, x...)

# @generated function _tex(A::T, y...) where {F,N,T<:AbstractArray{F,N}}
#     quote
#         sz = size(A)
#         s = zero(F)
        
#         Base.Cartesian.@nexprs $N i -> w_i = zero(F)
#         Base.Cartesian.@nloops $N i i -> begin
#             x = y[i] + one(F) / 2
#             px = unsafe_trunc(Int32, floor(x))
#             fx = fract(x)
#             px_i, w_i = px < 1 ? (1, fx) : begin
#                 px ≥ sz[i] ? (sz[i], fx) : (px, one(F)-fx)
#             end
#         end begin
#             w = prod(Base.Cartesian.@ntuple($N,w))
#             s += w * A[Base.Cartesian.@ntuple($N,px)...]
#         end
#         return s
#     end
# end

function _tex(A::T, x::Float64) where {T<:AbstractArray{Float64}}
    x += 0.5
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 1
        A[begin]
    elseif px ≥ sz[1]
        A[end]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end

function _tex(A::CuDeviceArray{Float32}, x::Float32)
    #x += 0.5f0
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 1
        A[begin]
    elseif px ≥ sz[1]
        A[end]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end

function _tex(A::MtlDeviceArray{Float32}, x::Float32)
    x += 0.5f0
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 1
        A[begin]
    elseif px ≥ sz[1]
        A[end]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end


function _tex(A::T, x::Float32) where {T<:CuDeviceArray{Float32}}
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 1
        A[begin]
    elseif px ≥ sz[1]
        A[end]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end

function _tex(A::T, x::Float64, y::Float64) where {T<:AbstractArray{Float64}}
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    px = max(1, min(sz[1], unsafe_trunc(Int64, px_float)))
    py = max(1, min(sz[2], unsafe_trunc(Int64, py_float)))
    (one(y) - fy) * ((one(x)-fx) * A[px,py] + fx * A[px+1,py]) +
    fy * ((one(x)-fx) * A[px,py+1] + fx * A[px+1,py+1])
end

function _tex(A::T, x::Float64, y::Float64, z::Float64) where {T<:AbstractArray{Float64}}
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    fz, pz_float = (fract(z), floor(z))
    px = max(1, min(sz[1], unsafe_trunc(Int64, px_float)))
    py = max(1, min(sz[2], unsafe_trunc(Int64, py_float)))
    pz = max(1, min(sz[3], unsafe_trunc(Int64, pz_float)))
    (one(z) - fz) *  ((one(y) - fy) *
                      ((one(x)-fx) * A[px,py,pz] + fx * A[px+1,py,pz]) +
                      fy *
                      ((one(x)-fx) * A[px,py+1,pz] + fx * A[px+1,py+1,pz])) +
    fz *  ((one(y) - fy) *
           ((one(x)-fx) * A[px,py,pz] + fx * A[px+1,py,pz+1]) +
           fy *
           ((one(x)-fx) * A[px,py+1,pz+1] + fx * A[px+1,py+1,pz+1]))
end

# function _tex(A::T, x::Float32) where {T<:AbstractArray{Float32}}
#     x += 0.5f0
#     sz = size(A)
#     fx  = fract(x)
#     px = unsafe_trunc(Int64, floor(x))
#     if px < 1
#         A[begin]
#     elseif px ≥ sz[1]
#         A[end]
#     else
#         (one(x)-fx) * A[px] + fx * A[px+1]
#     end
# end

function _tex(A::T, x::Float32, y::Float32) where {T<:AbstractArray{Float32}}
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    px = max(1, min(sz[1], unsafe_trunc(Int32, px_float)))
    py = max(1, min(sz[2], unsafe_trunc(Int32, py_float)))
    (one(y) - fy) * ((one(x)-fx) * A[px,py] + fx * A[px+1,py]) +
    fy * ((one(x)-fx) * A[px,py+1] + fx * A[px+1,py+1])
end

function _tex(A::T, x::Float32, y::Float32, z::Float32) where {T<:AbstractArray{Float32}}
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    fz, pz_float = (fract(z), floor(z))
    px = max(1, min(sz[1], unsafe_trunc(Int32, px_float)))
    py = max(1, min(sz[2], unsafe_trunc(Int32, py_float)))
    pz = max(1, min(sz[3], unsafe_trunc(Int32, pz_float)))
    (one(z) - fz) *  ((one(y) - fy) *
                      ((one(x)-fx) * A[px,py,pz] + fx * A[px+1,py,pz]) +
                      fy *
                      ((one(x)-fx) * A[px,py+1,pz] + fx * A[px+1,py+1,pz])) +
    fz *  ((one(y) - fy) *
           ((one(x)-fx) * A[px,py,pz] + fx * A[px+1,py,pz+1]) +
           fy *
           ((one(x)-fx) * A[px,py+1,pz+1] + fx * A[px+1,py+1,pz+1]))
end

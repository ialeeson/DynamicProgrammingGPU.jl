CUDA.tex(bc, A::Union{Array,CuDeviceArray,MtlDeviceArray}, x...) =
    _tex(bc, A, x...)

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

function _tex(::Clamp, A::T, x::Float64) where {T<:AbstractArray{Float64}}
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

function _tex(::Mirror, A::T, x::Float64) where {T<:AbstractArray{Float64}}
    x += 0.5
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 0
        A[-(px-1)]
    elseif 0 ≤ px < 1
        A[begin]
    elseif sz[1] ≤ px < sz[1] + 1
        A[end]
    elseif sz[1] + 1 ≤ px
        A[sz[1]-rem(px,sz[1])]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end


function _tex(::Mirror, A::Union{CuDeviceArray{Float32}, MtlDeviceArray{Float32}}, x::Float32)
    x += 0.5f0
    sz = size(A)
    fx  = fract(x)
    px = unsafe_trunc(Int64, floor(x))
    if px < 0
        A[-(px-1)]
    elseif 0 ≤ px < 1
        A[begin]
    elseif sz[1] ≤ px < sz[1] + 1
        A[end]
    elseif sz[1] + 1 ≤ px
        A[sz[1]-rem(px,sz[1])]
    else
        (one(x)-fx) * A[px] + fx * A[px+1]
    end
end


function _tex(::Mirror, A::CuDeviceArray{Float32}, x::Float32, y::Float32)
    
    x += 0.5f0
    y += 0.5f0
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    px = unsafe_trunc(Int64, px_float)
    py = unsafe_trunc(Int64, py_float)

    if px < 1
        if py < 1
            A[begin,begin]
        elseif py ≥ sz[2]
            A[begin,end]
        else
            (one(y)-fy) * A[1,py] + fy * A[1,py+1]
        end
    elseif px ≥ sz[1]
        if py < 1
            A[end,begin]
        elseif py ≥ sz[2]
            A[end,end]
        else
            (one(y)-fy) * A[end,py] + fy * A[end,py+1]
        end
    else
        if py < 1
            (one(x)-fx) * A[px,begin] + fx * A[px+1,begin]
        elseif py ≥ sz[2]
            (one(x)-fx) * A[px,end] + fx * A[px+1,end]
        else
            (one(y) - fy) * ((one(x)-fx) * A[px,py] + fx * A[px+1,py]) +
            fy * ((one(x)-fx) * A[px,py+1] + fx * A[px+1,py+1])
        end
    end
    
end

function _tex(::Clamp, A::Union{CuDeviceArray{Float32}, MtlDeviceArray{Float32}}, x::Float32)
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


# function _tex(A::T, x::Float32) where {T<:CuDeviceArray{Float32}}
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

function _tex(A::T, x::Float64, y::Float64) where {T<:AbstractArray{Float64}}
    
    x += 0.5
    y += 0.5
    sz = size(A)
    fx, px_float = (fract(x), floor(x))
    fy, py_float = (fract(y), floor(y))
    px = unsafe_trunc(Int64, px_float)
    py = unsafe_trunc(Int64, py_float)

    if px < 1
        if py < 1
            A[begin,begin]
        elseif py ≥ sz[2]
            A[begin,end]
        else
            (one(y)-fy) * A[1,py] + fy * A[1,py+1]
        end
    elseif px ≥ sz[1]
        if py < 1
            A[end,begin]
        elseif py ≥ sz[2]
            A[end,end]
        else
            (one(y)-fy) * A[end,py] + fy * A[end,py+1]
        end
    else
        if py < 1
            (one(x)-fx) * A[px,begin] + fx * A[px+1,begin]
        elseif py ≥ sz[2]
            (one(x)-fx) * A[px,end] + fx * A[px+1,end]
        else
            (one(y) - fy) * ((one(x)-fx) * A[px,py] + fx * A[px+1,py]) +
            fy * ((one(x)-fx) * A[px,py+1] + fx * A[px+1,py+1])
        end
    end
    
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

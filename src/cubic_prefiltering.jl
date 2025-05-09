function InitialCausalCoefficient(α, ::Clamp, c)
    (zn,s) = (one(α), zero(α))
    for i in 1:10^2
        s += zn * c[begin]
        zn *= α
    end
    return s
end
function InitialCausalCoefficient(α, ::Mirror, c)
    (zn,s) = (α, c[begin])
    for i in 1:min(length(c),12)
        s += zn * c[i]
        zn *= α
    end
    return s
end

function InitialAntiCausalCoefficient(α, ::Clamp, c)
    (zn,s) = (one(α), zero(α))
    for i in 1:10^2
        s += zn * c[end]
        zn *= α
    end
    return s - c[end]
end
TransmittedAntiCausalCoefficient(α, ::Clamp, c) = zero(α)

InitialAntiCausalCoefficient(α, ::Mirror, s) = zero(α)
TransmittedAntiCausalCoefficient(α, ::Mirror, c) = α * c[end]

prefilter!(bc, coeffs::AbstractArray{F}) where {F} = _prefilter!(bc, coeffs)
function _prefilter!(bc, coeffs::AbstractArray{F}) where {F}
    α = sqrt(3*one(F))-2
    
    ### Causal
    c = InitialCausalCoefficient(α, bc, coeffs)
    c_anti = InitialAntiCausalCoefficient(α, bc, coeffs)
    coeffs[begin], c_prev = (c,c)
    for i in 2:length(coeffs)
        c = coeffs[i] + α * c_prev
        coeffs[i], c_prev = (c,c)
    end

    ### AntiCausal
    c_trans = TransmittedAntiCausalCoefficient(α, bc, coeffs)
    λ = 3 * 2 * 1
    c = α * inv(α^2 - 1) * (c + c_anti + c_trans)
    coeffs[end], c_prev = (λ * c,c)
    for i in length(coeffs)-1:-1:1
        c = α * (c_prev - coeffs[i])
        coeffs[i], c_prev = (λ * c,c)
    end
end

function prefilter!(bc, coeffs::A) where {F, A<:AbstractArray{F,1}}
    sz, backend = (size(coeffs), get_backend(coeffs))
    prefilter_1d_x(backend, 1)(bc, coeffs, ndrange=1)
    synchronize(backend)
end

@kernel function prefilter_1d_x(bc, c::A) where {F, A<:AbstractArray{F,1}}
    # thread = @index(Local)
    # group = @index(Group)
    # group_sz = prod(@groupsize())
    #idx = thread + group_sz * (group-1)
    #α = sqrt(3)-2
    # λ = (1 - α) * (1 - inv(α))
    # sz = size(c)
    _prefilter!(bc, c)#, sz[1], 1, 1)
end

function prefilter!(bc, coeffs::A) where {F, A<:AbstractArray{F,2}}
    sz, backend = (size(coeffs), get_backend(coeffs))
    prefilter_2d_x(backend, min(sz[1], 1024))(bc, coeffs, ndrange=min(sz[1], 1024))
    synchronize(backend)
    prefilter_2d_y(backend, min(sz[2], 1024))(bc, coeffs, ndrange=min(sz[2], 1024))
    synchronize(backend)
end

@kernel function prefilter_2d_x(bc, c::A) where {F, A<:AbstractArray{F,2}}
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    stride = group_sz
    idx = thread + group_sz * (group-1)
    α = sqrt(3)-2
    λ = (1 - α) * (1 - inv(α))
    sz = size(c)
    for i in 1:cld(sz[1], group_sz)
        _prefilter!(bc, c, sz[2], sz[2], idx)
        idx += stride
    end
end

@kernel function prefilter_2d_y(bc, c::A) where {F, A<:AbstractArray{F,2}}
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    stride = prod(@groupsize())
    idx = thread + group_sz * (group-1)
    α = sqrt(3*one(F))-2
    λ = (1 - α) * (1 - inv(α))
    sz = size(c)
    for i in 1:cld(sz[2], group_sz)
        _prefilter!(bc, c, sz[1], 1, 1+(idx-1)*sz[1])
        idx += stride
    end
end

# function prefilter_2d(bc, c)
#     sz = size(c)
#     for i in axes(c,1)
#         _prefilter!(view(c, i, :))
#     end
#     for j in axes(c,2)
#         _prefilter!(view(c, :, j))
#     end
# end



# @kernel function prefilter_3d_x(c)
#     thread = @index(Local)
#     group = @index(group)
#     groupsize = @groupsize()[group]
#     stride = prod(@groupsize())
#     idx = thread + groupsize * (group-1)
#     α = sqrt(3*one(F))-2
#     λ = (1 - α) * (1 - inv(α))
#     sz = size(c)
#     for offset in idx-1:stride:sz[1]-1
#         prefilter!(c, sz[2]*sz[3], sz[2]*sz[3], offset)
#     end
# end

# @kernel function prefilter_3d_y(c)
#     thread = @index(Local)
#     group = @index(group)
#     groupsize = @groupsize()[group]
#     stride = prod(@groupsize())
#     idx = thread + groupsize * (group-1)
#     α = sqrt(3*one(F))-2
#     λ = (1 - α) * (1 - inv(α))
#     sz = size(c)
#     for offset in idx-1:stride:sz[2]-1
#         for zidx in 1:sz[3]
#             prefilter!(c, sz[1], 1, offset*sz[1] + sz[2])
#         end
#     end
# end

# @kernel function prefilter_3d_z(c)
#     thread = @index(Local)
#     group = @index(group)
#     groupsize = @groupsize()[group]
#     stride = prod(@groupsize())
#     idx = thread + groupsize * (group-1)
#     α = sqrt(3*one(F))-2
#     λ = (1 - α) * (1 - inv(α))
#     sz = size(c)
#     for offset in idx-1:stride:sz[2]-1
#         prefilter!(c, sz[1]*sz[2], sz[1]*sz[2], offset*sz[1]*sz[2])
#     end
# end

# function InitialCausalCoefficient(c::A, n, skip, start) where {F, A<:AbstractArray{F}}
#     α = sqrt(3*one(F))-2
#     (zn,s) = (α, c[1])
#     idx = start
#     for i in 1:n
# 	s += zn * c[idx]
# 	zn *= α
#         idx += skip
#     end
#     return s
# end

# prefilter!(coeffs::Vector) = prefilter_1d(coeffs)
# prefilter!(coeffs::Matrix) = prefilter_2d(coeffs)

# function prefilter!(coeffs::A) where {F, A<:AbstractArray{F,1}}
#     sz, backend = (size(coeffs), get_backend(coeffs))
#     prefilter_1d_x(backend, 1)(coeffs, ndrange=1)
#     synchronize(backend)
# end

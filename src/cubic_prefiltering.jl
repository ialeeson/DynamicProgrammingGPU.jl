@kernel function prefilter_1d_x(c::A) where {F, A<:AbstractArray{F,1}}
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    idx = thread + group_sz * (group-1)
    Pole = sqrt(3)-2
    λ = (1 - Pole) * (1 - inv(Pole))
    sz = size(c)
    _prefilter!(c, sz[1], 1, 1)
end

@kernel function prefilter_2d_x(c::A) where {F, A<:AbstractArray{F,2}}
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    stride = group_sz
    idx = thread + group_sz * (group-1)
    Pole = sqrt(3)-2
    λ = (1 - Pole) * (1 - inv(Pole))
    sz = size(c)
    for i in 1:cld(sz[1], group_sz)
        _prefilter!(c, sz[2], sz[2], idx)
        idx += stride
    end
end

@kernel function prefilter_2d_y(c::A) where {F, A<:AbstractArray{F,2}}
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    stride = prod(@groupsize())
    idx = thread + group_sz * (group-1)
    Pole = sqrt(3*one(F))-2
    λ = (1 - Pole) * (1 - inv(Pole))
    sz = size(c)
    for i in 1:cld(sz[2], group_sz)
        _prefilter!(c, sz[1], 1, 1+(idx-1)*sz[1])
        idx += stride
    end
end

# @kernel function prefilter_3d_x(c)
#     thread = @index(Local)
#     group = @index(group)
#     groupsize = @groupsize()[group]
#     stride = prod(@groupsize())
#     idx = thread + groupsize * (group-1)
#     Pole = sqrt(3*one(F))-2
#     λ = (1 - Pole) * (1 - inv(Pole))
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
#     Pole = sqrt(3*one(F))-2
#     λ = (1 - Pole) * (1 - inv(Pole))
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
#     Pole = sqrt(3*one(F))-2
#     λ = (1 - Pole) * (1 - inv(Pole))
#     sz = size(c)
#     for offset in idx-1:stride:sz[2]-1
#         prefilter!(c, sz[1]*sz[2], sz[1]*sz[2], offset*sz[1]*sz[2])
#     end
# end

function InitialCausalCoefficient(c::A, n, skip, start) where {F, A<:AbstractArray{F}}
    Pole = sqrt(3*one(F))-2
    (zn,s) = (Pole, c[1])
    idx = start
    for i in 1:n
	s += zn * c[idx]
	zn *= Pole
        idx += skip
    end
    return s
end

prefilter!(coeffs::Vector) = _prefilter!(coeffs, length(coeffs), 1, 1)

function prefilter!(coeffs::A) where {F, A<:AbstractArray{F,1}}
    sz, backend = (size(coeffs), get_backend(coeffs))
    prefilter_1d_x(backend, 1)(coeffs, ndrange=1)
    synchronize(backend)
end

function prefilter!(coeffs::A) where {F, A<:AbstractArray{F,2}}
    sz, backend = (size(coeffs), get_backend(coeffs))
    prefilter_2d_x(backend, min(sz[1], 1024))(coeffs, ndrange=min(sz[1], 1024))
    synchronize(backend)
    prefilter_2d_y(backend, min(sz[2], 1024))(coeffs, ndrange=min(sz[2], 1024))
    synchronize(backend)
end

function _prefilter!(coeffs::A, n, skip, start) where {F, A<:AbstractArray{F}}
    Pole = sqrt(3*one(F))-2
    λ = (1 - Pole) * (1 - inv(Pole))

    ### Causal
    c = λ * InitialCausalCoefficient(coeffs, n, skip, start)
    idx = start
    coeffs[idx], c_prev = (c,c)
    for i in 2:n
        idx += skip
        c = λ * coeffs[idx] + Pole * c_prev
        coeffs[idx], c_prev = (c,c)
    end

    ### AntiCausal
    c = Pole / (Pole - 1) * coeffs[idx]
    coeffs[idx], c_prev = (c,c)
    for i in n-1:-1:1
        idx -= skip
        c = Pole * (c_prev - coeffs[idx])
        coeffs[idx], c_prev = (c,c)
    end
end

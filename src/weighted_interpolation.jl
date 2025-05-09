struct PrecomputeInterpolation{O,A,B,C}
    itp::A
    weights::B
    tmp::C
    order::Val{O}
end
init(m::Markov{N}, order, sz) where {N} =
    ntuple(N) do i
        PrecomputeInterpolation(
            Interpolation(sz, order),
            m.weights[i],
            zeros(sz...),
            order
        )
    end
interpolate(t::PrecomputeInterpolation, x::Vararg{F,N}) where {F,N} =
    interpolate(t.itp, x...)

function copyto!(itp::NTuple{N,PrecomputeInterpolation}, layers, v) where {N}
    l, n = (length(layers), length(size(v)))
    for i in 1:N
        copyto!(itp[i], v)
    end
end
function copyto!(w::PrecomputeInterpolation, v)
    dev = get_backend(w.tmp)
    axis_tensor(dev, w.tmp, v, w.weights)
    synchronize(dev)
    copyto!(w.itp, w.tmp)
    synchronize(dev)
end
axis_tensor(dev::CPU, dest, src, weights) =
    axis_tensor_cpu(dest, src, weights)
axis_tensor(dev::Union{MetalBackend, CUDABackend}, dest, src, weights) =
    axis_tensor_gpu(dev)(dest, src, weights, ndrange=size(dest))

struct InPlaceInterpolation{O,A,B,C}
    itp::Interpolation{O,A}
    weights::B
    tmp::C
end

function copyto!(itp::NTuple{N,InPlaceInterpolation}, layers, v) where {N}
    l, n = (length(layers), length(size(v)))
    for (i,A) in enumerate(eachslice(v, dims=(n-l+1:n...,)))
        copyto!(itp[i], A)
    end
end
function copyto!(w::InPlaceInterpolation, v)
    copyto!(w.tmp, v)
    copyto!(w.itp, w.tmp)
end

init(quad::Quadrature{N}, order, sz) where {N} =
    ntuple(N) do i
        InPlaceInterpolation(
            Interpolation(sz, order),
            quad.weights[i],
            zeros(sz...)
        )
    end

function interpolate(t::InPlaceInterpolation{O}, x::Vararg{F,N}) where {O,F,N}
    s = zero(F)
    for j in eachindex(t.weights)
        w = t.weights[j]
        s += w * interpolate(
            t.itp,
            x...
        )
    end
    s
end

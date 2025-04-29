abstract type ModelParameters end

Adapt.adapt_structure(to::Union{Metal.MtlArrayAdaptor, CUDA.CuArrayKernelAdaptor}, p::T) where {T<:ModelParameters} = Base.typename(T).wrapper((
    adapt_to_gpu(to, getproperty(p, name)) for name in fieldnames(T))...
)

adapt_to_gpu(to, x::I) where {I<:Integer} = convert(Int32, x)
adapt_to_gpu(to, x::F) where {F<:AbstractFloat} = convert(Float32, x)
adapt_to_gpu(to, x::Tuple{Vararg{F}}) where {F<:AbstractFloat} =
    convert(Tuple{Vararg{Float32}}, x)
adapt_to_gpu(to, x) = Adapt.adapt(to, x)

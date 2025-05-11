using KernelAbstractions, BenchmarkTools
import KernelAbstractions.synchronize
import DynamicProgrammingGPU.prefilter!

@kernel function _prefilter!(res, A)

    tmp = @localmem Float32 1024*2
    thread = @index(Local)
    group = @index(Group)
    group_sz = prod(@groupsize())
    offset = (group - 1) * length(tmp)
    k = cld(length(tmp), group_sz)
    start = 1 + (thread - 1) * k
    n = min(8,k+1)
    
    α = sqrt(3.0f0) - 2.0f0
    λ = 3*2*1
    
    for i in thread:group_sz:length(tmp)
        tmp[i] = A[i + offset]
    end

    @synchronize()

    c = begin
        (z, s) = (α, tmp[start])
        ax = start == 1 ?
             (start:1:start+(n-1)) :
             (start:-1:start-(n-1))
        for i in ax
            s += z * tmp[i]
            z *= α
        end
        s
    end
    c_anti = begin
        (z, s) = (α, zero(α))
        ax = start+k-1 == length(tmp) ?
             (start+k-1:-1:start+k-1-(n-1)) :
             (start+k:1:start+k-1+(n-1))
        for i in ax
            s += z * tmp[i]
            z *= α
        end
        s
    end
    
    @synchronize()
    
    tmp[start], c_prev = (c,c)
    for i in start+1:start+k-1
        c = tmp[i] + α * c_prev
        tmp[i], c_prev = (c, c)
    end

    c = α * inv(α^2 - 1) * (c + c_anti)
    c_prev = c
    tmp[start+k-1] = λ * c
    for i in start+k-2:-1:start
        c = α * (c_prev - tmp[i])
        tmp[i], c_prev = (λ * c, c)
    end

    @synchronize()
    
    for i in thread:group_sz:length(tmp)
        res[i + offset] = tmp[i]
    end
    
end

function prefilter!(::Mirror, A::MtlArray)
    res = mtl(zeros(size(A)))
    wk_sz = 256
    _prefilter!(get_backend(A), wk_sz)(
        res, A, ndrange=cld(prod(size(A)), 1024*2) * wk_sz
    )
    synchronize(get_backend(A))
    res
end

function test(n...)
    A = collect(range.(zeros(length(n)), ones(length(n)), n)...)
    mtlA = mtl(convert(Array{Float32}, A))
    @info isapprox(
    #return hcat(
        Array(DynamicProgrammingGPU.prefilter!(Mirror(), mtlA)),
        DynamicProgrammingGPU.prefilter!(Mirror(), A)
    )
    #t0 = @belapsed DynamicProgrammingGPU.prefilter!(Mirror(), $mtlA)
    t1 = @belapsed DynamicProgrammingGPU.prefilter!(Mirror(), $A)
    #(t0, t1)
end

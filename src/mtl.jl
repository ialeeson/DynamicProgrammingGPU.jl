using KernelAbstractions, Metal

# 1d
@kernel function fkernel(f, u, v, l, r, tol)
    idx = @index(Global)
    tol = 
    u[idx], v[idx] = golden_section(f, l, r, tol)
end

function golden_section(f,l::Float64, r::Float64, tol::Float64)
    
    ϕ = 0.5 * (1.0 + sqrt(5.0))
    a₁ = ϕ - 1.0
    a₂ = 1.0 - a₁
    #n = log(tol/(r-l)) / log(a₁)
    
    x₁, x₄ = (l,r)
    x₂ = (x₄ + ϕ * x₁) / (1.0 + ϕ)
    x₃ = x₁ + x₄ - x₂
    fx₁, fx₂, fx₃, fx₄ = (f(x₁), f(x₂), f(x₃), f(x₄))

    for i in 1:10^2
        if min(fx₁,fx₂) < min(fx₃,fx₄)
            x₃, x₄ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₂, fx₃, fx₄ = (f(x₂), f(x₃), fx₃)
        else
            x₁, x₂ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₁, fx₂, fx₃  = (fx₂, f(x₂), f(x₃))
        end
        x₄-x₁ < tol && break
    end
    x = (x₁+x₄) * 0.5
    return x, f(x)
    
end

function golden_section(f,l,r,tol)
    
    ϕ = 0.5f0 * (1.0f0 + sqrt(5.0f0))
    a₁ = ϕ - 1.0f0
    a₂ = 1.0f0 - a₁
    
    x₁, x₄ = (l,r)
    x₂ = (x₄ + ϕ * x₁) / (1.0f0 + ϕ)
    x₃ = x₁ + x₄ - x₂
    fx₁, fx₂, fx₃, fx₄ = (f(x₁), f(x₂), f(x₃), f(x₄))

    for i in 1:10^2
        if min(fx₁,fx₂) < min(fx₃,fx₄)
            x₃, x₄ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0f0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₂, fx₃, fx₄ = (f(x₂), f(x₃), fx₃)
        else
            x₁, x₂ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0f0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₁, fx₂, fx₃  = (fx₂, f(x₂), f(x₃))
        end
        x₄-x₁ < tol && break
    end
    return (x₁+x₄) * 0.5f0
    
end

# m = MtlArray(rand(Float32,10^6))
# backend = get_backend(m)
# f = x->x
# fkernel(backend)(f, m, -10f0, 10f0, ndrange=length(m))
# @benchmark fkernel(get_backend(m))(f, m, -10f0, 10f0, ndrang

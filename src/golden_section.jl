struct GoldenSection
end

solve(solver::GoldenSection, f, lb, ub, args) =
    golden_section(f, lb, ub, args)

function golden_section(f::F, l::Float64, r::Float64, args) where {F}

    ϕ = 0.5 * (1.0 + sqrt(5.0))
    a₁ = ϕ - 1.0
    a₂ = 1.0 - a₁

    x₁ = l
    x₄ = r
    x₂ = (r + ϕ * l) / (1.0 + ϕ)
    x₃ = x₁ + x₄ - x₂
    fx₁ = f(x₁, args...)
    fx₂ = f(x₂, args...)
    fx₃ = f(x₃, args...)
    fx₄ = f(x₄, args...)
    
    tol = 1e-10
    n = 50#unsafe_trunc(Int64, log(tol/(x₄-x₁)) / log(a₁))

    i = 1
    while i < n && x₄-x₁ > tol
        if max(fx₁,fx₂) > max(fx₃,fx₄)
            x₃, x₄ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₂, fx₃, fx₄ = (f(x₂,args...), f(x₃,args...), fx₃)
        else
            x₁, x₂ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₁, fx₂, fx₃  = (fx₂, f(x₂,args...), f(x₃,args...))
        end
        i += 1
    end
    x = (x₁+x₄) * 0.5
    return x, f(x, args...)
    
end

function golden_section(f::F, l::Float32, r::Float32, args) where {F}
    
    ϕ = 0.5f0 * (1.0f0 + sqrt(5.0f0))
    a₁ = ϕ - 1.0f0
    a₂ = 1.0f0 - a₁
    
    x₁, x₄ = (l,r)
    x₂ = (r + ϕ * l) / (1f0 + ϕ)
    x₃ = x₁ + x₄ - x₂
    fx₁ = f(x₁, args...)
    fx₂ = f(x₂, args...)
    fx₃ = f(x₃, args...)
    fx₄ = f(x₄, args...)

    tol = 1f-5
    n = 50#unsafe_trunc(Int32, log(tol/(x₄-x₁)) / log(a₁))

    i = 1
    while i < n && x₄-x₁ > tol
        if max(fx₁,fx₂) > max(fx₃,fx₄)
            x₃, x₄ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0f0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₂, fx₃, fx₄ = (f(x₂,args...), f(x₃,args...), fx₃)
        else
            x₁, x₂ = (x₂, x₃)
            x₂ = (x₄ + ϕ * x₁) / (1.0f0 + ϕ)
            x₃ = x₄ - x₂ + x₁
            fx₁, fx₂, fx₃  = (fx₂, f(x₂,args...), f(x₃,args...))
        end
        i += 1
    end
    x = (x₁+x₄) * 0.5f0
    return x, f(x,args...)
    
end

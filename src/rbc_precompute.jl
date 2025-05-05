module RBCPrecompute

using DynamicProgrammingGPU

@kwdef struct Parameters{F} <: ModelParameters
    β::F = 0.984
    δ::F = 0.01
    α::F = 0.35
    γ::F = 1.2
    ρ::F = 0.95
    σ::F = 0.005
    min::F = 1e-5
    max::F = 10.0
end

c(u,s,p) = exp(s[2])*s[1]^p.α + (one(s[1])-p.δ)*s[1] - u[1]
f(u,s,v,p) = utility(c(u,s,p),p) + p.β * v(u[1])
bounds(b,s,v,p) = (
    zero(s[1]),
    exp(s[2])*s[1]^p.α + (one(s[1])-p.δ)*s[1] - 1f-5
)
utility(c,p) = c^(1-p.γ)/(1-p.γ)
v0(s,p) = utility(s[1],p)

function init(p, n; m=3)
    
    grid = Grid((p.min,), (p.max,), (n[1],))
    prob = ValueFunction(
        UnivariateOptimizationProblem(
            f,
            bounds,
            GoldenSection(),
        ),
        Val(3),
        MarkovIdentity(n[2])
    )
    DynamicProgrammingGPU.init(prob, grid, Base.Fix2(v0, p))
    
end

end

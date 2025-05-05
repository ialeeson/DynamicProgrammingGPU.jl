module CakeEatingProblem

using DynamicProgrammingGPU

@kwdef struct Parameters{F} <: ModelParameters
    β::F = 0.95
    γ::F = 1.2
    min::F = 1e-5
    max::F = 1e2
end

utility(c,p) = c^(1-p.γ)/(1-p.γ)
f(u,s,v,p) = utility(s[1]-u[1], p) + p.β * v(u[1])
bounds(b,s,v,p) = (zero(s[1]), s[1])
v0(s,p) = utility(s[1],p)

function init(p, n)

    grid = Grid((p.min,), (p.max,), n)
    prob = ValueFunction(
        UnivariateOptimizationProblem(
            f,
            bounds,
            GoldenSection()
        ),
        Val(3),
    )
    DynamicProgrammingGPU.init(prob, grid, Base.Fix2(v0,p))

end

end

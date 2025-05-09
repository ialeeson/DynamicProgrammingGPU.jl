using Interpolations
#include("DynamicProgrammingGPU.jl")
using DynamicProgrammingGPU
function test(first, last, n; m=10)
    g = Grid(first, last, n)
    g1 = Grid(first, last, n .* m .- 1)
    f = x->(prod(x))^(1-1.2)/(1-1.2)
    A = map(f, Base.product(range.(first, last, n)...))
    itp0 = Interpolation(Val(3), Mirror(), zeros(n...))
    sitp = DynamicProgrammingGPU.ScaledInterpolation(itp0, g)
    copyto!(itp0, copy(A))
    itp = extrapolate(scale(interpolate(copy(A), BSpline(Cubic(Reflect(OnGrid())))), range.(g.first, g.last, g.n)), Throw())
    # s = 0.0
    # sum(abs2(sitp(x...)- A[i])
    #     for (i,x) in enumerate(Base.product(range.(g.first, g.last, g.n)...)))
    [(sitp(x...), f(x...))
    for (i,x) in enumerate(Base.product(range.(g1.first, g1.last, g1.n)...))]
end

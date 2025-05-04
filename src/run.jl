include("DynamicProgrammingGPU.jl")
using .DynamicProgrammingGPU
include("cake_eating_problem.jl")
using Metal, BenchmarkTools, DataFrames
import CommonSolve.solve!, CommonSolve.init

include("cake_eating_problem.jl")
import .CakeEatingProblem as CP
include("rbc.jl")
import .RBC as RBC

function push_time!(t, name, n, nsteps)
    cpu = @belapsed(solve!(x, p),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n)
        )
    )
    gpu = @belapsed(solve!(x_mtl, p_mtl),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n);
            p_mtl = mtl(p);
            x_mtl = mtl(x)
        )
    )
    push!(t, (; name, n, cpu, gpu))
end

t = DataFrame(name = Symbol[], n = Tuple[], cpu = Float64[], gpu = Float64[])
[push_time!(t, name, (n,), 10^2) for n in 32 .* 2 .^ (0:2), name in (:CP,)]
[push_time!(t, name, (n,m), 10^2) for n in 32 .* 2 .^ (0:2), m in (8,), name in (:RBC,)]
t

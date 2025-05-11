using CUDA, BenchmarkTools, DataFrames, Adapt, FileIO, Dates, CSV, Metal
import CommonSolve.solve!, CommonSolve.init

# include("DynamicProgrammingGPU.jl")
# using .DynamicProgrammingGPU
# include("cake_eating_problem.jl")
# import .CakeEatingProblem as CP
# include("rbc.jl")
# import .RBC as RBC
# include("rbc_precompute.jl")
# import .RBCPrecompute

using DynamicProgrammingGPU
import DynamicProgrammingExamples.CakeEatingProblem as CP
import DynamicProgrammingExamples.RBC
import DynamicProgrammingExamples.RBCPrecompute

function push_time!(filename, t, date, name, n, nsteps)
    println("name: $name, n: $n, nsteps: $nsteps")
    cpu = @belapsed(solve!(x, p),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n)
        )
    )
    println("CPU: $cpu")
    gpu = @belapsed(solve!(x_cuda, p_cuda),
        setup = (
            p = getproperty(eval($name), :Parameters)();
            x = getproperty(eval($name), :init)(p,$n);
            p_cuda = mtl(p);
            x_cuda = mtl(x)
        )
    )
    println("GPU: $gpu")
    # tex = @belapsed(solve!(x_tex, p_tex),
    #     setup = (
    #         p = getproperty(eval($name), :Parameters)();
    #         x = getproperty(eval($name), :init)(p,$n);
    #         p_tex = cutex(p);
    #         x_tex = cutex(x)
    #     )
    # )
    push!(t, (; date, name, n=string(n), cpu, gpu, tex=NaN))
    CSV.write(filename, t; promote=true)
end

filename, date = ("times.csv", now())
types = Dict(1 => DateTime, 2 => Symbol, 3 => String, 4 => Float64, 5 => Float64, 6 => Float64)
t = isfile(filename) ? CSV.read(filename, DataFrame; types) : DataFrame(
    date = DateTime[], name = Symbol[], n = String[],
    cpu = Float64[], gpu = Float64[], tex = Float64[]
)
[push_time!(filename, t, date, name, (n,), 10^2) for n in 32 .* 2 .^ (12,),
    name in (:CP,)]
# [push_time!(filename, t, date, name, (n,m), 10^2) for n in 32 .* 2 .^ (3,5,7), m in (8,), name in (:RBC,)]
# [push_time!(filename, t, date, name, (n,m), 10^2) for n in 32 .* 2 .^ (3,5,7), m in (8,), name in (:RBCPrecompute,)]
# t

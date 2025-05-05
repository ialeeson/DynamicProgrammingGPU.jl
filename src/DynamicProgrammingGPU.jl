module DynamicProgrammingGPU

using KernelAbstractions, StaticArrays, Adapt
using Distributions
using Metal, CUDA
import KernelAbstractions.synchronize
import CUDA.tex, CUDA.unpack
import CommonSolve.init, CommonSolve.solve!, CommonSolve.solve
import Base.copyto!

include("grid.jl")
include("interpolation.jl")
include("linear_interpolation.jl")
include("cubic_interpolation.jl")
include("cubic_prefiltering.jl")
include("quadrature.jl")
include("markov.jl")
include("weighted_interpolation.jl")
include("layered_interpolation.jl")
include("solvers.jl")
include("golden_section.jl")
include("tex.jl")
include("rounding.jl")
include("value_function_iteration.jl")
include("euler_equation_iteration.jl")
include("kernels.jl")
include("parameters.jl")

Adapt.@adapt_structure ValueFunctionCache
Adapt.@adapt_structure LayeredValueFunctionCache
Adapt.@adapt_structure Interpolation
Adapt.@adapt_structure InPlaceInterpolation
Adapt.@adapt_structure PrecomputeInterpolation
Adapt.@adapt_structure Quadrature

export Grid, Interpolation
export ValueFunction, EulerEquation
export GoldenSection
export Tauchen, Quadrature, MarkovIdentity
export UnivariateNonlinearProblem, UnivariateOptimizationProblem
export NonlinearProblem
export ModelParameters

end

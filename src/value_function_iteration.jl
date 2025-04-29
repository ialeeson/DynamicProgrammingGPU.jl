struct ValueFunction{I,P,O}
    problem::P
    order::O
    integral::I
end
ValueFunction(p,o) = ValueFunction(p,o,missing)

function init(p::ValueFunction{I}, grid, v0) where {I<:QuadratureType}
    int = init(p.integral)
    g = Grid(
        (int.grid.first..., grid.first...),
        (int.grid.last..., grid.last...),
        (int.grid.n..., grid.n...)
    )
    itp = init(int, p.order, grid)
    ValueFunctionCache(g, p, itp, v0)
end
init(p::ValueFunction{Missing}, grid, v0) =
    ValueFunctionCache(grid, p, Interpolation(; order=p.order, grid), v0)

# function init(problem::ValueFunction, grid::Grid{N,F}, v0) where {N,F}

#     n = grid.n
#     u = zeros(F,n)
#     v = reshape(
#         map(v0, Iterators.product(range.(grid.first, grid.last, grid.n)...)),
#         grid.n
#     )
#     itp = if problem.markov isa Missing
#         Interpolation(copy(v), problem.order, grid)
#     else
#         dim, int = problem.markov
#         MarkovCache(
#             LayeredInterpolation(
#                 ntuple(n[dim]) do i
#                     copy(selectdim(v,dim,i))
#                 end,
#                 problem.order,
#                 grid
#             ),
#             similar(v),
#             dim,
#             int()
#         )
#     end
#     copyto!(itp,v)
#     ValueFunctionCache(grid, problem, u, v, itp)
            
# end

struct ValueFunctionCache{N,F,A,I,P}
    grid::Grid{N,F}
    problem::P
    u::A
    v::A
    itp::I
end
function ValueFunctionCache(grid, problem, itp, v0)
    v = reshape(
        map(
            v0,
            Iterators.product(range.(grid.first, grid.last, grid.n)...)
        ),
        grid.n
    )
    u = similar(v)
    copyto!(itp, v)
    ValueFunctionCache(grid, problem, u, v, itp)
end

function solve!(vf::ValueFunctionCache, p; nsteps=1)
    
    dev = get_backend(vf.v)
    warp_sz = DynamicProgrammingGPU.warp_size(length(vf.grid.n))
    
    for step in 1:nsteps
        if dev isa CPU
            u_cpu(vf.problem.problem, vf.u, vf.v, vf.itp, vf.grid, p)
        #elseif dev isa MetalBackend && all(mod.(vf.grid.n, warp_sz) .== 0)
        else
            u_gpu_square(dev, min(1024, prod(vf.grid.n)))(vf.problem.problem,
                vf.u, vf.v, vf.itp, vf.grid, p, warp_sz,
                ndrange=min(1024, prod(vf.grid.n)))
        # elseif dev isa MetalBackend
        #     u_gpu_linear(dev)(vf.problem.problem,
        #         vf.u, vf.v, vf.itp, vf.grid, p, ndrange=vf.grid.n)
        # else
        #     error("Unsupported backend.")
        end            
        KernelAbstractions.synchronize(dev)
        copyto!(vf.itp, vf.v)
        KernelAbstractions.synchronize(dev)
    end
    return vf.u
    
end

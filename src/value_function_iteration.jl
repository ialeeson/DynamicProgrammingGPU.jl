struct ValueFunction{I,P,O}
    problem::P
    order::O
    integral::I
end
ValueFunction(p,o) = ValueFunction(p,o,missing)

function init(p::ValueFunction{I}, grid, v0) where {I<:QuadratureType}
    int = init(p.integral)
    g = Grid(
        (grid.first..., int.grid.first...),
        (grid.last..., int.grid.last...),
        (grid.n..., int.grid.n...)
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
        u!(dev, prod(vf.grid.n), warp_sz, vf.itp,
            vf.problem.problem, vf.u, vf.v, vf.grid, p)
        KernelAbstractions.synchronize(dev)
        copyto!(vf.itp, vf.v)
        KernelAbstractions.synchronize(dev)
    end
    return vf.u
    
end

u!(dev, sz, warp_sz, itp, args...) = u_cpu(itp, args...)

u!(dev, sz, warp_sz, itp::Interpolation{F,N,O,A}, args...) where {F,N,O,A<:Union{CuArray,MtlArray}} =
    u_gpu_square(dev, min(1024, sz))(itp, args..., warp_sz,
        ndrange=min(1024, sz))

u!(dev, sz, warp_sz, itp::Interpolation{F,N,O,A}, args...) where {F,N,O,A<:CuTexture} =
    u_gpu_square(dev, min(1024, sz))(itp, args..., warp_sz,
        ndrange=min(1024, sz))

u!(dev, sz, warp_sz, itp::WeightedInterpolation{M,N,F,O,A}, args...) where {M,N,F,O,A<:Union{CuArray,MtlArray}} =
    u_gpu_square(dev, min(1024, sz))(itp, args..., warp_sz,
        ndrange=min(1024, sz))

u!(dev, sz, warp_sz, itp::WeightedInterpolation{M,N,F,O,A}, args...) where {M,N,F,O,A<:CuTexture} =
    u_gpu_square(dev, min(1024, sz))(itp, args..., warp_sz,
        ndrange=min(1024, sz))

u!(dev, sz, warp_sz, itp::LayeredInterpolation{M,N,F,O,A}, args...) where {M,N,F,O,A<:CuArray} =
    u_gpu_linear(dev)(itp, args..., ndrange=sz)

u!(dev, sz, warp_sz, itp::LayeredInterpolation{M,N,F,O,A}, args...) where {M,N,F,O,A<:CuTexture} =
    u_gpu_square(dev, min(1024, sz))(itp, args..., warp_sz,
        ndrange=min(1024, sz))

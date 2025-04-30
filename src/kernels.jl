@kernel function u_gpu_linear(itp, problem, u, v, grid, @Const(p))
    
    cidx = Tuple(@index(Global, Cartesian))
    x  = grid.first .+ grid.step .* (cidx .- 1)
    u[cidx...], v[cidx...] = solve(problem, u[cidx...],
        (grid.first, grid.last), x, cidx, itp, p)
    
end

@kernel function u_gpu_square(itp, problem, u, v, @Const(grid), @Const(p),
    @Const(warp_sz))
    
    thread = @index(Local)
    group = @index(Group)
    sz = prod(grid.n)
    group_sz = prod(@groupsize())
    stride = group_sz
    idx = thread + (group-1) * group_sz
    bounds = (grid.first, grid.last)
    
    for i in 1:cld(sz, group_sz)

        block_idx = cld(idx, prod(warp_sz))
        block_cidx = lidx_to_cidx(block_idx, cld.(grid.n, warp_sz))
        cidx = lidx_to_cidx(idx - prod(warp_sz) * (block_idx - 1), warp_sz) .+
               (block_cidx .- 1) .* warp_sz
        x  = grid.first .+ grid.step .* (cidx .- 1)
        u0 = u[cidx...]
        u[cidx...], v[cidx...] = solve(problem, u0, bounds, x, cidx, itp, p)
        idx += stride
        
    end
    
end

@generated function u_cpu(itp, problem::P, u::Array{F,N}, v::Array{F,N}, grid::Grid{N,F}, p) where {P,F,N}
    
    quote
        bounds = (grid.first, grid.last)
        Base.Cartesian.@nexprs $N d -> (x_d = grid.first[d])
        Base.Cartesian.@nloops $N i u d -> nothing d -> (x_d += grid.step[d]) begin
            u0 = Base.Cartesian.@nref $N u i
            cidx = Base.Cartesian.@ntuple $N i
            s = Base.Cartesian.@ntuple $N x
            Base.Cartesian.@nref($N, u, i), Base.Cartesian.@nref($N, v, i) =
                solve(problem, u0, bounds, s, cidx, itp, p)
        end
    end
    
end

lidx_to_cidx(i, n) = _lidx_to_cidx(i, reverse(n))
function _lidx_to_cidx(i, n)
    n0, n1 = (first(n), Base.tail(n))
    m1 = prod(n1)
    i1 = cld(i, m1)
    (
        _lidx_to_cidx(
            i - (i1-1) * m1,
            n1
        )...,
        i1
    )
end
_lidx_to_cidx(i, n::NTuple{1}) = i

function warp_size(dims)
    if dims == 1
        (32,)
    elseif dims == 2
        (8,4)
    elseif dims == 3
        (4,4,2)
    elseif dims == 4
        (4,2,2,2)
    elseif dims == 5
        (2,2,2,2,2)
    else
        (2,2,2,2,2,ones(Int,dims-5)...)
    end
end

# @kernel function u_1d(problem::F, u, v, itp, @Const(grid), @Const(p)) where {F}
#     thread = @index(Local)
#     group = @index(Group)
#     stride = @groupsize()[1]
#     sz = @ndrange()
#     bounds = (grid.first, grid.last)
#     idx = thread + (group-1) * stride
#     x  = grid.first[1] + grid.step[1] * (idx-1)
#     xstep = grid.step[1] * stride
#     for i in idx:stride:sz[1]
#         u[i], v[i] = solve(problem, u[i], bounds, x, i, itp, p)
#         x += xstep
#     end
# end

# function u_2d(f, v, u, grid, args...)
#     thread = @index(Local)
#     group = @index(Group)
#     stride = @groupsize()[1]
#     sz = @ndrange()
#     bounds = (grid.first, grid.last)
#     idx = thread + (group-1) * stride
#     x  = grid.first[1] + grid.step[1] * (idx-1)
#     xstep = grid.step[1] * stride
#     for i in idx:stride:sz[1]
#         u0 = Base.Cartesian.@nref $N u i
#         cidx = Base.Cartesian.@ntuple $N i
#         s = Base.Cartesian.@ntuple $N x
#         u[i], v[i] = solve(problem, u[i], bounds, x, i, itp, p)
#         x += xstep
#     end
# end

# @kernel function fsquare(A, sq_size)
    
#     thread = @index(Local)
#     group = @index(Group)
#     group_sz = prod(@groupsize())
#     stride = prod(@ndrange()) * group_sz
#     idx = thread + (group-1) * group_sz

#     sz = grid.n
#     block_sz = cld.(sz, sq_size)
        
#     for i in idx:stride:length(A)

#         block_rem = sz .- div.(sz, sq_size)
#         block_idx = 
#         block_cidx = lidx_to_cidx(block_idx, block_sz)
        
#         cidx = ntuple(length(sq_size)) do i
#             i == length(sq_size) ? block_cidx[i] * sq_size[i] :
#             (block_idx[i]-1) * sq_size[i] + idx
#         end
#         u0 = u[cidx...]
#         u[cidx...], v[cidx...] = solve(problem, u0, bounds, x, cidx, itp, p)
#         x += xstep
#     end
# end


# # @kernel function u_1d(problem, u, v, itp, @Const(grid), @Const(p))
# #     idx = @index(Local)
# #     group = @index(Group)
# #     stride = @groupsize()[1]
# #     sz = @ndrange()
# #     bounds = (grid.first, grid.last)
# #     x  = grid.first[1]
# #     xstep = grid.step[1] * stride
# #     for i in idx:stride:sz[1]
# #         u[i], v[i] = solve(problem, view(u,:,i), bounds, x, (i,), itp, p)
# #         x += xstep
# #     end
# # end

# # function u_2d(f, v, u, grid, args...)
# #     thread = (threadIdx().x, threadIdx().y)
# #     stride = (blockDim().x, blockDim().y)
# #     sz = size(v)
# #     lb = (minimum(grid[1]), minimum(grid[2]))
# #     ub = (maximum(grid[1]), maximum(grid[2]))    
# #     for i in thread[1]:stride[1]:sz[1]
# #         for j in thread[2]:stride[2]:sz[2]
# #             s = (grid[1][i], grid[2][j])
# #             solve(f, view(v,:,i,j), view(u,:,i,j), s, lb, ub, args...)
# #         end
# #     end
# # end

# # function u_3d(f, v, u, grid, args...)
# #     thread = (threadIdx().x, threadIdx().y, threadIdx().z)
# #     stride = (blockDim().x, blockDim().y, blockDim().z)
# #     sz = size(v)
# #     lb = (minimum(grid[1]), minimum(grid[2]), minimum(grid[3]))
# #     ub = (maximum(grid[1]), maximum(grid[2]), maximum(grid[3]))
# #     for i in thread[1]:stride[1]:sz[1]
# #         for j in thread[2]:stride[2]:sz[2]
# #             for k in thread[3]:stride[3]:sz[3]
# #                 s = (grid[1][i], grid[2][j], grid[3][k])
# #                 solve(f, view(v,:,i,j,k), view(u,:,i,j,k), s, lb, ub, args...)
# #             end
# #         end
# #     end
# # end

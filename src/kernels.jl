@kernel function axis_tensor_cpu(res, src, weights)
    for cidx in CartesianIndices(src)
        idx = Tuple(cidx)
        s = 0.0
        for j in eachindex(weights)
            s += weights[j] * src[j, idx[2:end]...]
        end
        res[idx...] = s
    end
end

@kernel function axis_tensor_gpu(res, src, weights)
    idx = @index(Global, NTuple)
    s = 0.0f0
    for j in eachindex(weights)
        s += weights[j] * src[j, idx[2:end]...]
    end
    res[idx...] = s
end

function u_cpu(itp, problem, u, v, grid, p, xpad)

    sitp = ScaledInterpolation(itp, grid)
    for cidx in CartesianIndices(grid.n)
        x  = grid.first .+ grid.step .* (Tuple(cidx) .- 1)
        u[cidx], v[cidx] = solve(problem, u[cidx],
            (grid.first, grid.last), (xpad..., x...), sitp, p)
    end
    
end

function u_cpu(itp::Interpolation{O}, weights, problem, u, v, grid, p, xpad) where {O}

    sitp = ScaledInterpolation(
        InPlaceInterpolation(itp, weights),
        grid
    )
    for cidx in CartesianIndices(grid.n)
        x  = grid.first .+ grid.step .* (Tuple(cidx) .- 1)
        u[cidx], v[cidx] = solve(problem, u[cidx],
            (grid.first, grid.last), (xpad..., x...), sitp, p)
    end
    
end

@kernel function u_gpu(itp, problem, u, v, @Const(grid), @Const(p), @Const(xpad))
    
    cidx = @index(Global, NTuple)
    sitp = ScaledInterpolation(itp, grid)
    x  = grid.first .+ grid.step .* (cidx .- 1)
    u[cidx...], v[cidx...] = solve(problem, u[cidx...],
        (grid.first, grid.last), (xpad..., x...), sitp, p)
    
end

@kernel function u_gpu(itp, @Const(weights), problem, u, v, @Const(grid), @Const(p), @Const(xpad))
    
    cidx = @index(Global, NTuple)
    sitp = ScaledInterpolation(
        InPlaceInterpolation(itp, weights),
        grid
    )
    x  = grid.first .+ grid.step .* (cidx .- 1)
    u[cidx...], v[cidx...] = solve(problem, u[cidx...],
        (grid.first, grid.last), (xpad..., x...), sitp, p)
    
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

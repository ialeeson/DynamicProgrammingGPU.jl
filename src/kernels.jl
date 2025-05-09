function axis_tensor_cpu(res, src, weights)
    for cidx in CartesianIndices(res)
        idx = Tuple(cidx)
        s = 0.0
        for j in eachindex(weights)
            s += weights[j] * src[idx..., j]
        end
        res[idx...] = s
    end
end

@kernel function axis_tensor_gpu(res, @Const(src), @Const(weights))
    idx = @index(Global, NTuple)
    s = 0.0f0
    for j in eachindex(weights)
        s += weights[j] * src[idx..., j]
    end
    res[idx...] = s
end

function u_cpu(itp, problem, u, v, grid, p, ipad, xpad)

    sitp = ScaledInterpolation(itp, grid)
    for cidx in CartesianIndices(grid.n)
        idx = Tuple(cidx)
        x  = grid.first .+ grid.step .* (idx .- 1)
        u[idx..., ipad...], v[idx..., ipad...] = solve(problem, u[idx..., ipad...], (grid.first, grid.last), (x..., xpad...), sitp, p)
    end
    
end

@kernel function u_gpu(itp, problem, u, v, @Const(grid), @Const(p), @Const(ipad), @Const(xpad))
    
    idx = @index(Global, NTuple)
    sitp = ScaledInterpolation(itp, grid)
    x  = grid.first .+ grid.step .* (idx .- 1)
    u[idx..., ipad...], v[idx..., ipad...] = solve(problem, u[idx..., ipad...], (grid.first, grid.last), (x..., xpad...), sitp, p)
    
end

function v_cpu(itp, f, u, v, grid, p, ipad, xpad)
    
    sitp = ScaledInterpolation(itp, grid)
    for cidx in CartesianIndices(grid.n)
        idx = Tuple(cidx)
        sitp = ScaledInterpolation(itp, grid)
        x  = grid.first .+ grid.step .* (idx .- 1)
        v[idx..., ipad...] = f(u[idx..., ipad...], (x..., xpad...), sitp, p)
    end
    
end

@kernel function v_gpu(itp, f, u, v, @Const(grid), @Const(p), @Const(ipad), @Const(xpad))
    
    idx = @index(Global, NTuple)
    sitp = ScaledInterpolation(itp, grid)
    x  = grid.first .+ grid.step .* (idx .- 1)
    v[idx..., ipad...] = f(u[idx..., ipad...], (x..., xpad...), sitp, p)
    
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

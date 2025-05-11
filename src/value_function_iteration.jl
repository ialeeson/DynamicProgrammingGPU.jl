struct ValueFunction{I,P,O}
    problem::P
    order::O
    integral::I
end
ValueFunction(p,o) = ValueFunction(p,o,missing)

function init(p::ValueFunction, grid, v0)
    int = init(p.integral)
    v = map(
        v0,
        Iterators.product(
            range.(grid.first, grid.last, grid.n)...,
            range.(int.grid.first, int.grid.last, int.grid.n)...
        )
    )
    u = similar(v)
    itp = init(int, p.order, grid.n)
    copyto!(itp, int.grid.n, v)
    LayeredValueFunctionCache(int.grid, grid, p.problem, u, v, itp)
end

function init(p::ValueFunction{Missing}, grid, v0)
    v = reshape(
        map(
            v0,
            Iterators.product(range.(grid.first, grid.last, grid.n)...)
        ),
        grid.n
    )
    u, itp = (similar(v), Interpolation(grid.n, p.order))
    copyto!(itp, v)
    ValueFunctionCache(grid, p.problem, u, v, itp)
end

struct ValueFunctionCache{A,N,F,I,P}
    grid::Grid{N,F}
    problem::P
    u::A
    v::A
    itp::I
end
copy!(vf::ValueFunctionCache) = copyto!(vf.itp, vf.v)

struct LayeredValueFunctionCache{A,M,N,L,F,I,P}
    layers::Grid{L,F}
    grid::Grid{N,F}
    problem::P
    u::A
    v::A
    itp::NTuple{M,I}
end
function copy!(vf::LayeredValueFunctionCache)
    copyto!(vf.itp, vf.layers.n, vf.v)
end

function solve!(vf::Union{ValueFunctionCache, LayeredValueFunctionCache},
    p; nsteps=1)

    for step in 1:nsteps
        u!(vf, p)
        copy!(vf)
        for i in 1:10
            v!(vf, p)
            copy!(vf)
        end
    end
    return vf.u
    
end

function u!(vf::ValueFunctionCache, p)
    dev = get_backend(vf.v)
    _u!(dev, vf.grid.n, vf.itp, vf.problem, vf.u, vf.v, vf.grid, p, (), ())
    synchronize(dev)
end

function u!(vf::LayeredValueFunctionCache, p)
    dev = get_backend(vf.v)
    sz = vf.grid.n
    for (i,xpad) in enumerate(Base.product(range.(vf.layers.first, vf.layers.last, vf.layers.n)...))
        ipad = lidx_to_cidx(i, vf.layers.n)
        _u!(dev, sz, vf.itp[i], vf.problem, vf.u, vf.v, vf.grid, p, ipad, xpad)
    end
    synchronize(dev)
end

_u!(dev::CPU, sz, itp, args...) =
    u_cpu(itp, args...)
_u!(dev::Union{MetalBackend,CUDABackend}, sz, itp, args...) =
    u_gpu(dev)(itp, args..., ndrange=sz)

function v!(vf::ValueFunctionCache, p)
    dev = get_backend(vf.v)
    _v!(dev, vf.grid.n, vf.itp, vf.problem.f, vf.u, vf.v, vf.grid, p, (), ())
    synchronize(dev)
end

function v!(vf::LayeredValueFunctionCache, p)
    dev = get_backend(vf.v)
    sz = vf.grid.n
    for (i,xpad) in enumerate(Base.product(range.(vf.layers.first, vf.layers.last, vf.layers.n)...))
        ipad = lidx_to_cidx(i, vf.layers.n)
        _v!(dev, sz, vf.itp[i], vf.problem.f, vf.u, vf.v, vf.grid, p,
            ipad, xpad)
    end
    synchronize(dev)
end

_v!(dev::CPU, sz, itp, args...) =
    v_cpu(itp, args...)
_v!(dev::Union{MetalBackend,CUDABackend}, sz, itp, args...) =
    v_gpu(dev)(itp, args..., ndrange=sz)

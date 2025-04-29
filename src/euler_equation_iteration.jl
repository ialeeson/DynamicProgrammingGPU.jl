struct EulerEquation{P,O,M}
    problem::P
    order::O
    markov::M
end
EulerEquation(p, o) = EulerEquation(p, o, missing)

function init(problem::EulerEquation, grid)

    n = grid.n
    u, v, resid = (zeros(n), zeros(n), zeros(n))
    itp = if markov isa Missing
        Interpolation(zeros(n), problem.order, grid)
    else
        MarkovCache(
            LayeredInterpolation(
                ntuple(n[dim]) do i
                    zeros(n[begin:dim-1]..., n[dim+1:end]...)
                end,
                problem.order,
                grid
            ),
            copy(v),
            dim,
            int()
        )
    end
    EulerEquationCache(problem, u, v, itp)
            
end

struct EulerEquationCache{A,I,P}
    problem::P
    u::A
    v::A
    resid::A
    itp::I
end

function solve!(vf::EulerEquationCache, p; nsteps=1)
    dev = get_backend(vf.v)
    for step in 1:nsteps
        u_1d(dev)(vf.problem.problem, vf.u, vf.v, vf.itp, vf.grid, p,
            ndrange=prod(vf.grid.n))
        KernelAbstractions.synchronize(dev)
        copyto!(vf.itp, vf.v)
        KernelAbstractions.synchronize(dev)
    end
    return u
end

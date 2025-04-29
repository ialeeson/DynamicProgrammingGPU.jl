### UnivariateConstrainted
struct UnivariateOptimizationProblem{F,B,S}
    f::F
    bounds::B
    solver::S
end

function solve(problem::UnivariateOptimizationProblem, u0, bounds, args...)
    lb, ub = problem.bounds(bounds, args...)
    golden_section(problem.f, lb, ub, args)
    #solve(problem.solver, problem.f, lb, ub, args)
    # if ub < lb
         
    # else
    #     solve(problem.solver, problem.f, lb, ub, args)
    # end
end

### UnivariateNonlinear
struct UnivariateNonlinearProblem
    f
    bounds
    solver
end

function solve(problem::UnivariateNonlinearProblem, u0, extrema, args...)
    bounds = problem.bounds(extrema, args...)
    lb, ub = (bounds[2][1], bounds[1][1])
    if problem.f(lb, args...) < zero(u0)
        lb, zero(u0)
    elseif problem.f(ub, args...) > zero(u0)
        ub, zero(u0)
    else
        solve(problem.solver, problem.f, u0, args...)
    end        
end

### NonlinearProblem
struct NonlinearProblem
    f
    solver
end

solve(problem::NonlinearProblem, u0, extrema, args...) =
    solve(problem.solver, problem.f, u0, args...)

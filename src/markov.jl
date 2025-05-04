abstract type MarkovType end

struct Markov{N,F,A}
    grid::Grid{1,F}
    weights::NTuple{N,A}
end

struct MarkovIdentity{I} <: MarkovType
    n::I
end
function init(m::MarkovIdentity)
    grid = Grid((1.0,), (float(m.n),), (m.n,))
    weights = zeros(m.n,m.n)
    for i in 1:n
        weights[i,i] = 1.0
    end
    Markov(grid, weights)
end

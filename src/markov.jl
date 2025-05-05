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
    weights = ntuple(i->rand(m.n), m.n)
    for i in 1:m.n
        weights[i] ./= sum(weights[i])
    end
    Markov(grid, weights)
end

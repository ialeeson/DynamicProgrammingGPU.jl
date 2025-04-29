using ShiftedArrays

abstract type QuadratureType end

struct Quadrature{N,F,A}
    grid::Grid{1,F}
    weights::NTuple{N,A}
end

### Tauchen
struct Tauchen <: QuadratureType
    ρ
    σ
    m
    n
end
function init(t::Tauchen)
    
    nodes = range(-t.m, t.m, t.n) .* t.σ
    shift = (nodes[2] - nodes[1])/2
    dists = ntuple(i -> Normal(t.ρ * nodes[i], t.σ), t.n)
    weights = map(dists) do d
        ω_prev, w = (zero(t.ρ), zeros(eltype(t.ρ), t.n))
        for i in 1:t.n-1
            ω = cdf(d, nodes[i] + shift)
            w[i] = ω - ω_prev
            ω_prev = ω
        end
        w[end] = one(t.ρ)-sum(w)
        w
    end
    
    Quadrature(
        Grid(
            (first(nodes),),
            (last(nodes),),
            (length(nodes),)
        ),
        weights
    )
    
end


mutable struct DiscreteMDP <: MDP{Int64, Int64}
    T::Array{Float64, 3} # SPxAxS
    R::Matrix{Float64} # SxA
    discount::Float64
end

mutable struct DiscretePOMDP <: POMDP{Int64, Int64, Int64}
    T::Array{Float64, 3} # SPxAxS
    R::Matrix{Float64} # SxA
    O::Array{Float64, 3} # OxAxSP
    discount::Float64
end

const DiscreteProb = Union{DiscreteMDP, DiscretePOMDP}

# Distribution Type and methods
# XXX: this should be replaced with Categorical when https://github.com/JuliaStats/Distributions.jl/issues/743 is fixed
struct DiscreteDistribution{P<:AbstractVector{Float64}}
    p::P
end

support(d::DiscreteDistribution) = 1:length(p)

pdf(d::DiscreteDistribution, sp::Int64) = d.p[sp] # T(s', a, s)

rand(rng::AbstractRNG, d::DiscreteDistribution) = sample(rng, Weights(d.p))

# MDP and POMDP common methods

n_states(prob::DiscreteProb) = size(prob.T, 1)
n_actions(prob::DiscreteProb) = size(prob.T, 2)

states(p::DiscreteProb) = 1:n_states(p)
actions(p::DiscreteProb) = 1:n_actions(p)

discount(p::DiscreteProb) = p.discount

transition(p::DiscreteProb, s::Int64, a::Int64) = DiscreteDistribution(view(p.T, :, a, s))

reward(prob::DiscreteProb, s::Int64, a::Int64) = prob.R[s, a]

initialstate_distribution(p::DiscreteProb) = DiscreteDistribution(ones(n_states(p))./n_states(p))

# POMDP only methods
n_observations(p::DiscreteProb) = size(p.O)

observations(p::DiscretePOMDP) = 1:p.n_observations(p)

observation(p::DiscretePOMDP, a::Int64, sp::Int64) = DiscreteDistribution(view(p.O, :, a, sp))

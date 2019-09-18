@doc raw"""
    TabularMDP(T, R, discount)

Specify a discrete MDP in tabular form.

- The transition matrix is 3 dimensional of size ``|\mathcal{S}|\times |\mathcal{A}| \times |\mathcal{S}|``, then `T[sj, a, si]` corresponds to the probability of ending in `sj` while taking action `a` in `si`.
- The reward matrix is 2 dimensional of size ``|\mathcal{S}|\times |\mathcal{A}|``, where `R[s, a]` is the reward obtained when taking action `a` in state `s`.
"""
mutable struct TabularMDP <: MDP{Int64, Int64}
    T::Array{Float64, 3} # SPxAxS
    R::Matrix{Float64} # SxA
    discount::Float64
end

@doc raw"""
    TabularPOMDP(T, R, O, discount)

Specify a discrete POMDP in tabular form.

- The transition matrix is 3 dimensional of size ``|\mathcal{S}|\times |\mathcal{A}| \times |\mathcal{S}|``, then `T[sj, a, si]` corresponds to the probability of ending in `sj` while taking action `a` in `si`.
- The observation matrix is also 3 dimensional of size ``|\mathcal{O}| \times |\mathcal{A}| \times |\mathcal{S}|``, `O[o, a, sp]` represents the probability of observing `o` in in state `sp` and action `a`.
- The reward matrix is 2 dimensional of size ``|\mathcal{S}|\times |\mathcal{A}|``, where `R[s, a]` is the reward obtained when taking action `a` in state `s`.
"""
mutable struct TabularPOMDP <: POMDP{Int64, Int64, Int64}
    T::Array{Float64, 3} # SPxAxS
    R::Matrix{Float64} # SxA
    O::Array{Float64, 3} # OxAxSP
    discount::Float64
end

const TabularProblem = Union{TabularMDP, TabularPOMDP}

# Distribution Type and methods
# XXX: this should be replaced with Categorical when https://github.com/JuliaStats/Distributions.jl/issues/743 is fixed
struct DiscreteDistribution{P<:AbstractVector{Float64}}
    p::P
end

support(d::DiscreteDistribution) = 1:length(d.p)

pdf(d::DiscreteDistribution, sp::Int64) = d.p[sp] # T(s', a, s)

rand(rng::AbstractRNG, d::DiscreteDistribution) = sample(rng, Weights(d.p))

# MDP and POMDP common methods

n_states(prob::TabularProblem) = size(prob.T, 1)
n_actions(prob::TabularProblem) = size(prob.T, 2)

states(p::TabularProblem) = 1:n_states(p)
actions(p::TabularProblem) = 1:n_actions(p)

stateindex(::TabularProblem, s::Int64) = s
actionindex(::TabularProblem, a::Int64) = a

discount(p::TabularProblem) = p.discount

transition(p::TabularProblem, s::Int64, a::Int64) = DiscreteDistribution(view(p.T, :, a, s))

reward(prob::TabularProblem, s::Int64, a::Int64) = prob.R[s, a]

initialstate_distribution(p::TabularProblem) = DiscreteDistribution(ones(n_states(p))./n_states(p))

# POMDP only methods
n_observations(p::TabularProblem) = size(p.O, 1)

observations(p::TabularPOMDP) = 1:n_observations(p)

observation(p::TabularPOMDP, a::Int64, sp::Int64) = DiscreteDistribution(view(p.O, :, a, sp))

obsindex(p::TabularPOMDP, o::Int64) = o

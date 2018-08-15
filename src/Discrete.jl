using Distributions


mutable struct DiscreteMDP <: MDP{Int64, Int64}
    T::Array{Float64, 3} # SxAxS
    R::Matrix{Float64} # SxA
    ns::Int64
    na::Int64
    discount::Float64
end

function DiscreteMDP(T::Array{Float64, 3}, R::Matrix{Float64}, discount::Float64)
    # add some checks
    ns, na = size(R, 1), size(R, 2)
    return DiscreteMDP(T, R, ns, na, discount)
end

mutable struct DiscretePOMDP <: POMDP{Int64, Int64, Int64}
    T::Array{Float64, 3} # SxAxS
    R::Matrix{Float64} # SxA
    O::Array{Float64, 3} # OxAxS
    ns::Int64
    na::Int64
    no::Int64
    discount::Float64
end

function DiscretePOMDP(T::Array{Float64, 3}, R::Matrix{Float64}, O::Array{Float64, 3}, discount::Float64)
    # add some checks
    ns, na, no = size(R, 1), size(R, 2), size(O, 1)
    return DiscretePOMDP(T, R, O, ns, na, no)
end

const DiscreteProb = Union{DiscreteMDP, DiscretePOMDP}

# Distribution Type and methods

mutable struct DiscreteDistribution
    D::Array{Float64, 3}
    s::Int64
    a::Int64
    it::UnitRange{Int64}
end

iterator(d::DiscreteDistribution) = d.it

pdf(d::DiscreteDistribution, sp::Int64) = d.D[sp, d.a, d.s] # T(s', a, s)

function rand(rng::AbstractRNG, d::DiscreteDistribution)
    cat = Weights(d.D[:,d.a,d.s])
    return sample(rng, cat)
end

# Space  and methods

mutable struct DiscreteSpace
    it::UnitRange{Int64}
end

iterator(space::DiscreteSpace) = space.it

rand(rng::AbstractRNG, space::DiscreteSpace) = rand(rng, space.it)


# MDP and POMDP common methods

n_states(prob::DiscreteProb) = prob.ns
n_actions(prob::DiscreteProb) = prob.na

state_index(::DiscreteProb, s::Int64) = s
action_index(::DiscreteProb, a::Int64) = a

states(p::DiscreteProb) = DiscreteSpace(1:p.ns)
actions(p::DiscreteProb) = DiscreteSpace(1:p.na)

discount(p::DiscreteProb) = p.discount

function transition(prob::DiscreteProb, s::Int64, a::Int64)
    d::DiscreteDistribution=DiscreteDistribution(prob.T,0,0,1:prob.ns)
    d.s = s
    d.a = a
    return d
end

reward(prob::DiscreteProb, s::Int64, a::Int64) = prob.R[s, a]
reward(prob::DiscreteProb, s::Int64, a::Int64, sp::Int64) = prob.R[s, a]

mutable struct StateDist
    cat::Vector{Float64}
end
initial_state_distribution(prob::DiscreteProb) = StateDist(ones(prob.ns)/prob.ns)
rand(rng::AbstractRNG, d::StateDist) = sample(rng, Weights(d.cat))
pdf(d::StateDist, s::Int64) = d.cat[s]
iterator(d::StateDist) = collect(1:length(d.cat))

# POMDP only methods
observation_index(::DiscretePOMDP, o::Int64) = o

n_observations(prob::DiscreteProb) = prob.no

observations(p::DiscretePOMDP) = DiscreteSpace(1:p.no)

function observation(prob::DiscretePOMDP, a::Int64, sp::Int64)
    d=DiscreteDistribution(prob.O,0,0,1:prob.no)
    d.s = sp
    d.a = a
    return d
end

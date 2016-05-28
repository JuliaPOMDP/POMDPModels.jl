using Distributions


type DiscreteMDP <: MDP{Int64, Int64}
    T::Array{Float64, 3} # SxAxS
    R::Matrix{Float64} # SxA
    ns::Int64
    na::Int64
    discount::Float64
    function DiscreteMDP(T::Array{Float64, 3}, R::Matrix{Float64}, discount::Float64)
        mdp = new()
        # add some checks
        mdp.ns, mdp.na = size(R, 1), size(R, 2)
        mdp.discount = discount
        mdp.T, mdp.R = T, R
        return mdp
    end
end

type DiscretePOMDP <: POMDP{Int64, Int64, Int64}
    T::Array{Float64, 3} # SxAxS
    R::Matrix{Float64} # SxA
    O::Array{Float64, 3} # OxAxS
    ns::Int64
    na::Int64
    no::Int64
    discount::Float64
    function DiscretePOMDP(T::Array{Float64, 3}, R::Matrix{Float64}, O::Array{Float64, 3}, discount::Float64)
        pomdp = new()
        # add some checks
        pomdp.ns, pomdp.na, pomdp.no = size(R, 1), size(R, 2), size(O, 1)
        pomdp.discount = discount
        pomdp.T, pomdp.R, pomdp.O = T, R, O
        return pomdp
    end
end

typealias DiscreteProb Union{DiscreteMDP, DiscretePOMDP}

# Distribution Type and methods

type DiscreteDistribution <: AbstractDistribution
    D::Array{Float64, 3}
    s::Int64
    a::Int64
    it::UnitRange{Int64}
end

iterator(d::DiscreteDistribution) = d.it

pdf(d::DiscreteDistribution, sp::Int64) = d.D[sp, d.a, d.s] # T(s', a, s)

function rand(rng::AbstractRNG, d::DiscreteDistribution, s::Int64)
    cat = Categorical(d.D[:,d.a,d.s])
    rand(cat)
end

# Space Type and methods

type DiscreteSpace <: AbstractSpace
    it::UnitRange{Int64}
end

iterator(space::DiscreteSpace) = space.it

rand(rng::AbstractRNG, space::DiscreteSpace, sample::Int64) = rand(rng, space.it)


# MDP and POMDP common methods

n_states(prob::DiscreteProb) = prob.ns
n_actions(prob::DiscreteProb) = prob.na

create_state(::DiscreteProb) = zero(Int64)
create_action(::DiscreteProb) = zero(Int64)
state_index(::DiscreteProb, s::Int64) = s
action_index(::DiscreteProb, a::Int64) = a

states(p::DiscreteProb) = DiscreteSpace(1:p.ns)
actions(p::DiscreteProb) = DiscreteSpace(1:p.na)

discount(p::DiscreteProb) = p.discount

create_transition_distribution(prob::DiscreteProb) = DiscreteDistribution(prob.T, 0, 0, 1:prob.ns)

function transition(prob::DiscreteProb, s::Int64, a::Int64, d::DiscreteDistribution=DiscreteDistribution(prob.T,0,0,1:prob.ns))
    d.s = s
    d.a = a
    return d
end

reward(prob::DiscreteProb, s::Int64, a::Int64) = prob.R[s, a]
reward(prob::DiscreteProb, s::Int64, a::Int64, sp::Int64) = prob.R[s, a]

initial_state_distribution(prob::DiscreteProb) = Categorical(prob.ns)


# POMDP only methods

create_observation(::DiscretePOMDP) = zero(Int64)
observation_index(::DiscretePOMDP, o::Int64) = o

observations(p::DiscretePOMDP) = DiscreteSpace(1:p.no)

create_observation_distribution(prob::DiscretePOMDP) = DiscreteDistribution(prob.O, 0, 0, 1:prob.no)


function observation(prob::DiscretePOMDP, s::Int64, a::Int64, d::DiscreteDistribution=DiscreteDistribution(prob.O,0,0,1:prob.no))
    d.s = s
    d.a = a
    return d
end

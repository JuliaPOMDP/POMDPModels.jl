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

type DiscreteDistribution
    D::Array{Float64, 3}
    s::Int64
    a::Int64
    it::UnitRange{Int64}
end

iterator(d::DiscreteDistribution) = d.it

pdf(d::DiscreteDistribution, sp::Int64) = d.D[sp, d.a, d.s] # T(s', a, s)

function rand(rng::AbstractRNG, d::DiscreteDistribution)
    cat = WeightVec(d.D[:,d.a,d.s])
    return sample(rng, cat)
end

# Space Type and methods

type DiscreteSpace
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

type StateDist
    cat::Vector{Float64}
end
initial_state_distribution(prob::DiscreteProb) = StateDist(ones(prob.ns)/prob.ns)
rand(rng::AbstractRNG, d::StateDist) = sample(rng, WeightVec(d.cat))
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

Base.convert(::Type{Array{Float64}}, s::Int64, prob::Union{DiscreteMDP,DiscretePOMDP}) = Float64[s]
Base.convert(::Type{Int}, s::Array{Float64}, prob::Union{DiscreteMDP,DiscretePOMDP}) = Int(s[1])

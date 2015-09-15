#=
using POMDPs
using Distributions
using POMDPToolbox
=#

# TODO: this might be better implemented with immutable states, actions, and observations

type BabyPOMDP <: POMDP
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
end
BabyPOMDP(r_feed, r_hungry) = BabyPOMDP(r_feed, r_hungry, 0.1, 0.8, 0.1, 0.9)

type BabyState
    hungry::Bool
end
==(u::BabyState, v::BabyState) = u.hungry==v.hungry
hash(s::BabyState) = hash(s.hungry)

type BabyObservation 
    crying::Bool
end
==(u::BabyObservation, v::BabyObservation) = u.crying==v.crying
hash(o::BabyObservation) = hash(o.crying)

type BabyAction
    feed::Bool
end
==(u::BabyAction, v::BabyAction) = u.feed==v.feed
hash(a::BabyAction) = hash(a.feed)

type BabyStateDistribution <: Belief
    p_hungry::Float64 # probability of being hungry
end
BabyStateDistribution() = BabyStateDistribution(0.0)

type BabyObservationDistribution <: AbstractDistribution
    p_crying::Float64 # probability of crying
end
BabyObservationDistribution() = BabyObservationDistribution(0.0)


create_state(::BabyPOMDP) = BabyState(false)
create_observation(::BabyPOMDP) = BabyObservation(false)
create_action(::BabyPOMDP) = BabyAction(false)
create_transition_distribution(::BabyPOMDP) = BabyStateDistribution()
create_observation_distribution(::BabyPOMDP) = BabyObservationDistribution()
create_belief(::BabyPOMDP) = BabyStateDistribution()

n_states(::BabyPOMDP) = 2
n_actions(::BabyPOMDP) = 2
n_observations(::BabyPOMDP) = 2

function transition(pomdp::BabyPOMDP, s::BabyState, a::BabyAction, d::BabyStateDistribution=create_belief(pomdp))
    if !a.feed && s.hungry
        d.p_hungry = 1.0
    elseif a.feed 
        d.p_hungry = 0.0
    else
        d.p_hungry = pomdp.p_become_hungry
    end
    return d
end

function observation(pomdp::BabyPOMDP, s::BabyState, a::BabyAction, d::BabyObservationDistribution=create_observation_distribution(pomdp))
    if s.hungry
        d.p_crying = pomdp.p_cry_when_hungry
    else
        d.p_crying = pomdp.p_cry_when_not_hungry
    end
    return d
end

function reward(pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    r = 0.0
    if s.hungry
        r += pomdp.r_hungry
    end
    if a.feed
        r += pomdp.r_feed
    end
    return r
end

function rand!(rng::AbstractRNG, s::BabyState, d::BabyStateDistribution)
    s.hungry = (rand(rng) <= d.p_hungry)
    return s
end

function rand!(rng::AbstractRNG, o::BabyObservation, d::BabyObservationDistribution)
    o.crying = (rand(rng) <= d.p_crying)
    return o
end

function belief(p::BabyPOMDP, old::BabyStateDistribution, a::BabyAction, o::BabyObservation, b::BabyStateDistribution=create_belief(p))
    if a.feed
        b.p_hungry = 0.0
    else # did not feed
        b.p_hungry = old.p_hungry + (1.0-old.p_hungry)*p.p_become_hungry # this is from the system dynamics
        # bayes rule
        if o.crying
            b.p_hungry = (p.p_cry_when_hungry*b.p_hungry)/(p.p_cry_when_hungry*b.p_hungry + p.p_cry_when_not_hungry*(1.0-b.p_hungry))
        else # not crying
            b.p_hungry = ((1.0-p.p_cry_when_hungry)*b.p_hungry)/((1.0-p.p_cry_when_hungry)*b.p_hungry + (1.0-p.p_cry_when_not_hungry)*(1.0-b.p_hungry))
        end
    end
    return b
end

dimensions(::BabyObservationDistribution) = 1
dimensions(::BabyStateDistribution) = 1

function states(::BabyPOMDP)
    [BabyState(i) for i = 0:1]
end

# const ACTION_SET = [BabyAction(i) for i = 0:1]

function actions(::BabyPOMDP)
    return [BabyAction(i) for i in 0:1]
end

# # needs to be updated after interface changes
# function actions!(acts::Vector{BabyAction}, ::BabyPOMDP, s::BabyState)
#     acts[1:end] = [BabyAction(i) for i in 0:1] # ACTION_SET[1:end] 
# end

# # interpolants don't work for now because I got rid of using Distributions.Bernoulli [This is my (Zach's) fault]
# create_interpolants(::BabyPOMDP) = Interpolants()
# 
# function interpolants!(interpolants::Interpolants, d::BabyStateDistribution)
#     empty!(interpolants)
#     ph = params(d.ishungry)[1]
#     push!(interpolants, 1, (1-ph)) # hungry
#     push!(interpolants, 2, (ph)) # not hungry
#     interpolants
# end
# 
# function interpolants!(interpolants::Interpolants, d::BabyObservationDistribution)
#     empty!(interpolants)
#     ph = params(d.iscrying)[1]
#     push!(interpolants, 1, (1-ph)) # crying
#     push!(interpolants, 2, (ph)) # not crying
#     interpolants
# end
# 
# length(interps::Interpolants) = interps.length
# 
# weight(interps::Interpolants, i::Int64) = interps.weights[i]
# 
# index(interps::Interpolants, i::Int64) = interps.indices[i]

function convert!(x::Vector{Float64}, state::BabyState)
    x[1] = float(state.hungry)
    x
end

discount(p::BabyPOMDP) = p.discount
isterminal(::BabyState) = false

# some example policies
type Starve <: Policy
end
function action(::BabyPOMDP, ::Starve, ::Belief, a=BabyAction(false))
    a.feed = false
    return a # Never feed :(
end

type AlwaysFeed <: Policy
end
function action(::BabyPOMDP, ::AlwaysFeed, ::Belief, a=BabyAction(true))
    a.feed = true
    return a
end

# feed when the previous observation was crying - this is nearly optimal
type FeedWhenCrying <: Policy
end
function action(::BabyPOMDP, ::FeedWhenCrying, b::PreviousObservation, a=BabyAction(false))
    if b.observation == nothing || b.observation.crying == false
        a.feed = false
        return a
    else # is crying
        a.feed = true
        return a
    end
end
function action(::BabyPOMDP, ::FeedWhenCrying, b::BabyStateDistribution, a=BabyAction(false))
    a.feed = false
    return a  
end

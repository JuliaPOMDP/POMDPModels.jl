#=
using POMDPs
using Distributions
using POMDPToolbox
=#

# TODO implement update_belief!

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

type BabyObservation 
    crying::Bool
end

type BabyAction
    feed::Bool
end

type BabyStateDistribution <: Belief
    ishungry::Bernoulli

    BabyStateDistribution() = new(Bernoulli(0.2))
    BabyStateDistribution(p_ishungry::Float64) = new(Bernoulli(p_ishungry))
end

type BabyObservationDistribution <: AbstractDistribution
    iscrying::Bernoulli

    BabyObservationDistribution() = new(Bernoulli(0.1))
end


create_state(::BabyPOMDP) = BabyState(false)
create_observation(::BabyPOMDP) = BabyObservation(false)
create_transition_distribution(::BabyPOMDP) = BabyStateDistribution()
create_observation_distribution(::BabyPOMDP) = BabyObservationDistribution()

n_states(::BabyPOMDP) = 2
n_actions(::BabyPOMDP) = 2
n_observations(::BabyPOMDP) = 2

function transition!(d::BabyStateDistribution, pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    if !a.feed && s.hungry
        d.ishungry = Bernoulli(1.0)
    elseif a.feed 
        d.ishungry = Bernoulli(0.0)
    else
        d.ishungry = Bernoulli(pomdp.p_become_hungry)
    end
    d
end


function observation!(d::BabyObservationDistribution, pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    if s.hungry
        d.iscrying = Bernoulli(pomdp.p_cry_when_hungry)
    else
        d.iscrying = Bernoulli(pomdp.p_cry_when_not_hungry)
    end
    d
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
    # XXX does not use rng
    s.hungry = rand(d.ishungry)
    s
end

function rand!(rng::AbstractRNG, o::BabyObservation, d::BabyObservationDistribution)
    # XXX does not use rng
    o.crying = rand(d.iscrying)
    o
end

dimensions(::BabyObservationDistribution) = 1
dimensions(::BabyStateDistribution) = 1

function states(::BabyPOMDP)
    [BabyState(i) for i = 0:1]
end

const ACTION_SET = [BabyAction(i) for i = 0:1]

function actions!(acts::Vector{BabyAction}, ::BabyPOMDP, s::BabyState)
    acts[1:end] = ACTION_SET[1:end] 
end

create_interpolants(::BabyPOMDP) = Interpolants()

function interpolants!(interpolants::Interpolants, d::BabyStateDistribution)
    empty!(interpolants)
    ph = params(d.ishungry)[1]
    push!(interpolants, 1, (1-ph)) # hungry
    push!(interpolants, 2, (ph)) # not hungry
    interpolants
end

function interpolants!(interpolants::Interpolants, d::BabyObservationDistribution)
    empty!(interpolants)
    ph = params(d.iscrying)[1]
    push!(interpolants, 1, (1-ph)) # crying
    push!(interpolants, 2, (ph)) # not crying
    interpolants
end

length(interps::Interpolants) = interps.length

weight(interps::Interpolants, i::Int64) = interps.weights[i]

index(interps::Interpolants, i::Int64) = interps.indices[i]

function convert!(x::Vector{Float64}, state::BabyState)
    x[1] = float(state.hungry)
    x
end

discount(p::BabyPOMDP) = p.discount
isterminal(::BabyState) = false

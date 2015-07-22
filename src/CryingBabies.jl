#module CryingBabies

using POMDPs
using Distributions
using POMDPToolbox

import POMDPs.states
import POMDPs.actions!
import POMDPs.create_action
import POMDPs.create_state
import POMDPs.create_transition
import POMDPs.create_observation
import POMDPs.transition!
import POMDPs.observation!
import POMDPs.reward
import POMDPs.n_states
import POMDPs.n_actions
import POMDPs.n_observations
import POMDPs.rand!

import POMDPs: create_interpolants, interpolants!, weight, index
import POMDPs: dimensions

#=
export 
    BabyPOMDP,
    BabyState,
    BabyAction,
    BabyObservation,
    TransitionDistribution,
    ObservationDistribution,
    create_state,
    create_action,
    create_transition,
    create_observation,
    n_states,
    n_actions,
    n_observations,
    states,
    actions!,
    transition!
    observation!,
    reward,
    rand!,
    interpolants!
=#

type BabyPOMDP <: POMDP
    r_feed::Float64
    r_hungry::Float64
end

type BabyState
    hungry::Bool
end

type BabyObservation 
    crying::Bool
end

type BabyAction
    feed::Bool
end

type TransitionDistribution <: AbstractDistribution
    ishungry::Bernoulli

    TransitionDistribution() = new(Bernoulli(0.2))
end

type ObservationDistribution <: AbstractDistribution
    iscrying::Bernoulli

    ObservationDistribution() = new(Bernoulli(0.1))
end


create_state(::BabyPOMDP) = BabyState(false)
create_action(::BabyPOMDP) = BabyAction(false)
create_transition(::BabyPOMDP) = TransitionDistribution()
create_observation(::BabyPOMDP) = ObservationDistribution()

n_states(::BabyPOMDP) = 2
n_actions(::BabyPOMDP) = 2
n_observations(::BabyPOMDP) = 2

function transition!(d::TransitionDistribution, pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    if !a.feed && s.hungry
        d.ishungry = Bernoulli(1.0)
    elseif a.feed 
        d.ishungry = Bernoulli(0.0)
    else
        d.ishungry = Bernoulli(0.1)
    end
    d
end


function observation!(d::ObservationDistribution, pomdp::BabyPOMDP, s::BabyState, a::BabyAction)
    if s.hungry
        d.iscrying = Bernoulli(0.8)
    else
        d.iscrying = Bernoulli(0.1)
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

function rand!(s::BabyState, d::TransitionDistribution)
    s.hungry = rand(d.ishungry)
    s
end

function rand!(s::BabyState, d::ObservationDistribution)
    s.hungry = rand(d.crying)
    s
end

dimensions(::ObservationDistribution) = 1
dimensions(::TransitionDistribution) = 1

function states(::BabyPOMDP)
    [BabyState(i) for i = 0:1]
end

const ACTION_SET = [BabyAction(i) for i = 0:1]

function actions!(acts::Vector{BabyAction}, ::BabyPOMDP, s::BabyState)
    acts[1:end] = ACTION_SET[1:end] 
end

create_interpolants(::BabyPOMDP) = Interpolants()

function interpolants!(interpolants::Interpolants, d::TransitionDistribution)
    empty!(interpolants)
    ph = params(d.ishungry)[1]
    push!(interpolants, 1, (1-ph)) # hungry
    push!(interpolants, 2, (ph)) # not hungry
    interpolants
end

function interpolants!(interpolants::Interpolants, d::ObservationDistribution)
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


#end #mdoule

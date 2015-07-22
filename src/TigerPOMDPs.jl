module TigerPOMDPs

using POMDPs
using Distributions
using POMDPToolbox

import POMDPs: domain, states, actions, actions!, observations, observations!
import POMDPs: create_transition, create_observation
import POMDPs: reward, transition!, observation!
import POMDPs: n_states, n_actions, n_observations
import POMDPs: length, weight, index

export 
    TigerPOMDP,
    TigerState,
    TigerAction,
    TigerObservation,
    TransitionDistribution,
    ObservationDistribution,
    StateSpace,
    ActionSpace,
    ObsSpace,

    states,
    actions,
    actions!,
    observations,
    observations!,
    domain,
    n_states,
    n_actions,
    n_observations,

    create_transition,
    create_observation,
    transition!
    observation!,
    reward,

    length,
    weight,
    index


type TigerPOMDP <: POMDP
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
end

type TigerState
    tigerleft::Bool
end

type TigerObservation
    obsleft::Bool
end

# Incompatible until Julia 0.4: @enum TigerAction listen=1 openleft=2 openright=3

abstract Enum
immutable TigerAction <: Enum
    val::Int
    function TigerAction(i::Integer)
        @assert 1 <= i <= 3
        new(i)
    end
end

==(x::TigerAction, y::TigerAction) = x.val == y.val

const listen = TigerAction(1)
const openleft = TigerAction(2)
const openright = TigerAction(3)

abstract TigerDistribution <: AbstractDistribution

type TransitionDistribution <: TigerDistribution
    interps::Interpolants
end
function create_transition(::TigerPOMDP)
    interps = Interpolants(2)
    push!(interps, 1, 0.5)
    push!(interps, 2, 0.5)
    d = TransitionDistribution(interps)
    d
end

type ObservationDistribution <: TigerDistribution
    interps::Interpolants
end
function create_observation(::TigerPOMDP)
    interps = Interpolants(2)
    push!(interps, 1, 0.5)
    push!(interps, 2, 0.5)
    d = ObservationDistribution(interps)
    d
end

Base.length(d::TigerDistribution) = d.interps.length
weight(d::TigerDistribution, i::Int64) = d.interps.weights[i]
index(d::TigerDistribution, i::Int64) = d.interps.indices[i]

n_states(::TigerPOMDP) = 2
n_actions(::TigerPOMDP) = 3
n_observations(::TigerPOMDP) = 2


# Resets the problem after opening door; does nothing after listening
function transition!(d::TransitionDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    interps = d.interps
    if a == openleft || a == openright
        fill!(interps.weights, 0.5)    
    elseif s.tigerleft
        interps.weights[1] = 1.0
        interps.weights[2] = 0.0
    else
        interps.weights[1] = 0.0
        interps.weights[2] = 1.0
    end
    d
end

function observation!(d::ObservationDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    interps = d.interps
    if a == listen
        if s.tigerleft
            interps.weights[1] = 0.85 
            interps.weights[2] = 0.15 
        else
            interps.weights[1] = 0.15
            interps.weights[2] = 0.85 
        end
    else 
        fill!(interps.weights, 0.5)    
    end
    d
end

function reward(pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
    r = 0.0
    if a == listen
        r += pomdp.r_listen
    end
    if a == openleft
        if s.tigerleft
            r += pomdp.r_findtiger
        else
            r += pomdp.r_escapetiger
        end
    end
    if a == openright
        if s.tigerleft
            r += pomdp.r_escapetiger
        else
            r += pomdp.r_findtiger
        end
    end
    return r
end


abstract TigerSpace <: AbstractSpace

type StateSpace <: TigerSpace
    states::Vector{TigerState}
end
states(::TigerPOMDP) = StateSpace([TigerState(true), TigerState(false)])
domain(space::StateSpace) = space.states

type ActionSpace <: TigerSpace
    actions::Vector{TigerAction}
end
actions(::TigerPOMDP) = ActionSpace([listen, openleft, openright])
actions!(acts::ActionSpace, ::TigerPOMDP, s::TigerState) = acts
domain(space::ActionSpace) = space.actions

type ObsSpace <: TigerSpace
    obs::Vector{TigerObservation}
end
observations(::TigerPOMDP) = ObsSpace([TigerObservation(true), TigerState(false)])
observations!(obs::ObsSpace, ::TigerPOMDP, s::TigerState) = obs
domain(space::ObsSpace) = space.obs


end #module

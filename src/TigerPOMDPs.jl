type TigerPOMDP <: POMDP
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    discount_factor::Float64
end
function TigerPOMDP()
    return TigerPOMDP(-1.0, -100.0, 10.0, 0.95)
end

type TigerState
    tigerleft::Bool
end
create_state(::TigerPOMDP) = TigerState(false)

type TigerObservation
    obsleft::Bool
end
create_observation(::TigerPOMDP) = TigerObservation(false)

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

abstract AbstractTigerDistribution <: AbstractDistribution

type TigerStateDistribution <: AbstractTigerDistribution
    interps::Interpolants
end
function create_transition_distribution(::TigerPOMDP)
    interps = Interpolants(2)
    push!(interps, 1, 0.5)
    push!(interps, 2, 0.5)
    d = TigerStateDistribution(interps)
    d
end

type TigerObservationDistribution <: AbstractTigerDistribution
    interps::Interpolants
end
function create_observation_distribution(::TigerPOMDP)
    interps = Interpolants(2)
    push!(interps, 1, 0.5)
    push!(interps, 2, 0.5)
    d = TigerObservationDistribution(interps)
    d
end

Base.length(d::AbstractTigerDistribution) = d.interps.length
weight(d::AbstractTigerDistribution, i::Int64) = d.interps.weights[i]
index(d::AbstractTigerDistribution, i::Int64) = d.interps.indices[i]

n_states(::TigerPOMDP) = 2
n_actions(::TigerPOMDP) = 3
n_observations(::TigerPOMDP) = 2


# Resets the problem after opening door; does nothing after listening
function transition!(d::TigerStateDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
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

function observation!(d::TigerObservationDistribution, pomdp::TigerPOMDP, s::TigerState, a::TigerAction)
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


type TigerStateSpace 
    states::Vector{TigerState}
end
states(::TigerPOMDP) = TigerStateSpace([TigerState(true), TigerState(false)])
domain(space::TigerStateSpace) = space.states

type TigerActionSpace 
    actions::Vector{TigerAction}
end
actions(::TigerPOMDP) = TigerActionSpace([listen, openleft, openright])
actions!(acts::TigerActionSpace, ::TigerPOMDP, s::TigerState) = acts
domain(space::TigerActionSpace) = space.actions

type TigerObservationSpace 
    obs::Vector{TigerObservation}
end
observations(::TigerPOMDP) = TigerObservationSpace([TigerObservation(true), TigerState(false)])
observations!(obs::TigerObservationSpace, ::TigerPOMDP, s::TigerState) = obs
domain(space::TigerObservationSpace) = space.obs

function rand!(rng::AbstractRNG, s::TigerState, d::TigerStateDistribution)
    c = Categorical(d.interps.weights)     
    sp = d.interps.indices[rand(c)]
    sp == 1 ? (s.tigerleft=true) : (s.tigerleft=false)
    s
end

function rand!(rng::AbstractRNG, o::TigerObservation, d::TigerObservationDistribution)
    c = Categorical(d.interps.weights)     
    op = d.interps.indices[rand(c)]
    op == 1 ? (o.obsleft=true) : (o.obsleft=false)
    o
end

discount(pomdp::TigerPOMDP) = pomdp.discount_factor

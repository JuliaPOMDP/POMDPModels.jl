type TigerPOMDP <: POMDP{Bool, Int64, Bool}
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    p_listen_correctly::Float64
    discount_factor::Float64
    vec_state::Vector{Float64}
end
TigerPOMDP() = TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95, zeros(1))

create_state(::TigerPOMDP) = zero(Bool)
create_observation(::TigerPOMDP) = zero(Bool)
create_action(::TigerPOMDP) = zero(Int64)
state_index(::TigerPOMDP, s::Bool) = Int64(s) + 1
action_index(::TigerPOMDP, a::Int) = a+1

create_belief(::TigerPOMDP) = DiscreteBelief(2)
initial_belief(::TigerPOMDP) = DiscreteBelief(2)

const listen = 0
const openleft = 1
const openright = 2

type TigerDistribution <: AbstractDistribution
    p::Float64
    it::Vector{Bool}
end
TigerDistribution() = TigerDistribution(0.5, [true, false])
iterator(d::TigerDistribution) = d.it

create_transition_distribution(::TigerPOMDP) = TigerDistribution()
create_observation_distribution(::TigerPOMDP) = TigerDistribution()

#Base.length(d::AbstractTigerDistribution) = d.interps.length
#weight(d::AbstractTigerDistribution, i::Int64) = d.interps.weights[i]
#index(d::AbstractTigerDistribution, i::Int64) = d.interps.indices[i]

function pdf(d::TigerDistribution, so::Bool)
    so ? (return d.p) : (return 1.0-d.p)
end

rand(rng::AbstractRNG, d::TigerDistribution, s::Bool) = rand(rng) <= d.p
rand(rng::AbstractRNG, d::TigerDistribution) = rand(rng) <= d.p

n_states(::TigerPOMDP) = 2
n_actions(::TigerPOMDP) = 3
n_observations(::TigerPOMDP) = 2

# Resets the problem after opening door; does nothing after listening
function transition(pomdp::TigerPOMDP, s::Bool, a::Int64, d::TigerDistribution=create_transition_distribution(pomdp))
    if a == 1 || a == 2
        d.p = 0.5
    elseif s
        d.p = 1.0
    else
        d.p = 0.0
    end
    d
end

function observation(pomdp::TigerPOMDP, a::Int64, sp::Bool, d::TigerDistribution=create_observation_distribution(pomdp))
    pc = pomdp.p_listen_correctly
    if a == 0
        sp ? (d.p = pc) : (d.p = 1.0-pc)
    else 
        d.p = 0.5
    end
    d
end

function observation(pomdp::TigerPOMDP, s::Bool, a::Int64, sp::Bool, d::TigerDistribution=create_observation_distribution(pomdp))
    return observation(pomdp, a, sp, d)
end


function reward(pomdp::TigerPOMDP, s::Bool, a::Int64)
    r = 0.0
    a == 0 ? (r+=pomdp.r_listen) : (nothing)
    if a == 1
        s ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == 2
        s ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end
reward(pomdp::TigerPOMDP, s::Bool, a::Int64, sp::Bool) = reward(pomdp, s, a)


initial_state_distribution(pomdp::TigerPOMDP) = TigerDistribution(0.5, [true, false])     


type TigerStateSpace <: AbstractSpace
    states::Vector{Bool}
end
states(::TigerPOMDP) = TigerStateSpace([true, false])
iterator(space::TigerStateSpace) = space.states
dimensions(::TigerStateSpace) = 1
function rand(rng::AbstractRNG, space::TigerStateSpace, s::Bool)
    p = rand(rng)
    p > 0.5 ? (return true) : (return false)
end

type TigerActionSpace <: AbstractSpace
    actions::Vector{Int64}
end
actions(::TigerPOMDP) = TigerActionSpace([0,1,2])
actions(pomdp::TigerPOMDP, s::Bool, acts::TigerActionSpace=actions(pomdp)) = acts
iterator(space::TigerActionSpace) = space.actions
dimensions(::TigerActionSpace) = 1
function rand(rng::AbstractRNG, space::TigerActionSpace, a::Int64)
    a = rand(rng, 0:2)
    return a
end

type TigerObservationSpace <: AbstractSpace
    obs::Vector{Bool}
end
observations(::TigerPOMDP) = TigerObservationSpace([true, false])
observations(::TigerPOMDP, s::Bool, obs::TigerObservationSpace=observations(pomdp)) = obs
iterator(space::TigerObservationSpace) = space.obs
dimensions(::TigerObservationSpace) = 1

function upperbound(pomdp::TigerPOMDP, s::Bool)
    return pomdp.r_escapetiger 
end

discount(pomdp::TigerPOMDP) = pomdp.discount_factor


function generate_o(p::TigerPOMDP, s::Bool, rng::AbstractRNG, o::Bool=create_observation(p))
    d = observation(p, create_action(p), s) # obs distrubtion not action dependant
    return rand(rng, d)
end

# same for both state and observation
function vec(p::TigerPOMDP, so::Bool) 
    p.vec_state[1] = so
    return p.vec_state
end

type TigerBeliefUpdater <: Updater{TigerDistribution}
    pomdp::TigerPOMDP
end


function update(bu::TigerBeliefUpdater, bold::DiscreteBelief, a::Int64, o::Bool, b::DiscreteBelief=create_belief(bu.pomdp))
    bl = bold[1]
    br = bold[2]
    p = bu.pomdp.p_listen_correctly
    if a == 0
        if o
            bl *= p
            br *= (1.0-p)
        else
            bl *= (1.0-p)
            br *= p
        end
    else
        bl = 0.5
        br = 0.5
    end
    norm = bl+br
    b[1] = bl / norm
    b[2] = br / norm
    b
end




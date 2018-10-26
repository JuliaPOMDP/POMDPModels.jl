mutable struct TigerPOMDP <: POMDP{Bool, Int64, Bool}
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    p_listen_correctly::Float64
    discount_factor::Float64
end
TigerPOMDP() = TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)

states(::TigerPOMDP) = (true, false)
observations(::TigerPOMDP) = (true, false)

stateindex(::TigerPOMDP, s::Bool) = Int64(s) + 1
actionindex(::TigerPOMDP, a::Int) = a + 1
obsindex(::TigerPOMDP, o::Bool) = Int64(o) + 1

initial_belief(::TigerPOMDP) = DiscreteBelief(2)

const TIGER_LISTEN = 0
const TIGER_OPEN_LEFT = 1
const TIGER_OPEN_RIGHT = 2

const TIGER_LEFT = true
const TIGER_RIGHT = false


n_states(::TigerPOMDP) = 2
n_actions(::TigerPOMDP) = 3
n_observations(::TigerPOMDP) = 2

# Resets the problem after opening door; does nothing after listening
function transition(pomdp::TigerPOMDP, s::Bool, a::Int64)
    p = 1.0
    if a == 1 || a == 2
        p = 0.5
    elseif s
        p = 1.0
    else
        p = 0.0
    end
    return BoolDistribution(p)
end

function observation(pomdp::TigerPOMDP, a::Int64, sp::Bool)
    pc = pomdp.p_listen_correctly
    p = 1.0
    if a == 0
        sp ? (p = pc) : (p = 1.0-pc)
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function observation(pomdp::TigerPOMDP, s::Bool, a::Int64, sp::Bool)
    return observation(pomdp, a, sp)
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


initialstate_distribution(pomdp::TigerPOMDP) = BoolDistribution(0.5)

actions(::TigerPOMDP) = [0,1,2]

function upperbound(pomdp::TigerPOMDP, s::Bool)
    return pomdp.r_escapetiger
end

discount(pomdp::TigerPOMDP) = pomdp.discount_factor

function generate_o(p::TigerPOMDP, s::Bool, rng::AbstractRNG)
    d = observation(p, 0, s) # obs distrubtion not action dependant
    return rand(rng, d)
end

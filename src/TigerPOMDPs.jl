mutable struct TigerPOMDP <: POMDP{Bool, Int64, Bool}
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    p_listen_correctly::Float64
    discount_factor::Float64
end
TigerPOMDP() = TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)

states(::TigerPOMDP) = (false, true)
observations(::TigerPOMDP) = (false, true)

stateindex(::TigerPOMDP, s::Bool) = Int64(s) + 1
actionindex(::TigerPOMDP, a::Int) = a + 1
obsindex(::TigerPOMDP, o::Bool) = Int64(o) + 1

initial_belief(::TigerPOMDP) = DiscreteBelief(2)

const TIGER_LISTEN = 0
const TIGER_OPEN_LEFT = 1
const TIGER_OPEN_RIGHT = 2

const TIGER_LEFT = false
const TIGER_RIGHT = true


# Resets the problem after opening door; does nothing after listening
function transition(pomdp::TigerPOMDP, s::Bool, a::Int64)
    if a == TIGER_OPEN_LEFT || a == TIGER_OPEN_RIGHT
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
    if a == TIGER_LISTEN
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
    a == TIGER_LISTEN && (r+=pomdp.r_listen)
    if a == TIGER_OPEN_LEFT
        s == TIGER_LEFT ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == TIGER_OPEN_RIGHT
        s == TIGER_RIGHT ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    return r
end
reward(pomdp::TigerPOMDP, s::Bool, a::Int64, sp::Bool) = reward(pomdp, s, a)


initialstate(pomdp::TigerPOMDP) = BoolDistribution(0.5)

actions(::TigerPOMDP) = 0:2

function upperbound(pomdp::TigerPOMDP, s::Bool)
    return pomdp.r_escapetiger
end

discount(pomdp::TigerPOMDP) = pomdp.discount_factor

initialobs(p::TigerPOMDP, s::Bool) = observation(p, 0, s) # listen 

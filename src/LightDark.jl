# A one-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg

import Base: ==, +, *, -

"""
    LightDark1DState

## Fields
- `y`: position
- `status`: 0 = normal, negative = terminal
"""
struct LightDark1DState
    status::Int64
    y::Float64
end

*(n::Number, s::LightDark1DState) = LightDark1DState(s.status, n*s.y)

"""
    LightDark1D

A one-dimensional light dark problem. The goal is to be near 0. Observations are noisy measurements of the position.

Model
-----

   -3-2-1 0 1 2 3
...| | | | | | | | ...
          G   S

Here G is the goal. S is the starting location
"""
mutable struct LightDark1D{F<:Function} <: POMDPs.POMDP{LightDark1DState,Int,Float64}
    discount_factor::Float64
    correct_r::Float64
    incorrect_r::Float64
    step_size::Float64
    movement_cost::Float64
    sigma::F
end

default_sigma(x::Float64) = abs(x - 5)/sqrt(2) + 1e-2

LightDark1D() = LightDark1D(0.9, 10.0, -10.0, 1.0, 0.0, default_sigma)

discount(p::LightDark1D) = p.discount_factor

isterminal(::LightDark1D, act::Int64) = act == 0

isterminal(::LightDark1D, s::LightDark1DState) = s.status < 0


actions(::LightDark1D) = -1:1
n_actions(p::LightDark1D) = length(actions(p))


struct LDNormalStateDist
    mean::Float64
    std::Float64
end

sampletype(::Type{LDNormalStateDist}) = LightDark1DState 
rand(rng::AbstractRNG, d::LDNormalStateDist) = LightDark1DState(0, d.mean + randn(rng)*d.std) 
initialstate_distribution(pomdp::LightDark1D) = LDNormalStateDist(2, 3)

observation(p::LightDark1D, sp::LightDark1DState) = Normal(sp.y, p.sigma(sp.y))

function transition(p::LightDark1D, s::LightDark1DState, a::Int)
    if a == 0
        return Deterministic(LightDark1DState(-1, s.y+a*p.step_size))
    else
        return Deterministic(LightDark1DState(s.status, s.y+a*p.step_size))
    end
end

function reward(p::LightDark1D, s::LightDark1DState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if abs(s.y) < 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost*a
    end
end


convert_s(::Type{A}, s::LightDark1DState, p::LightDark1D) where A<:AbstractArray = eltype(A)[s.status, s.y]
convert_s(::Type{LightDark1DState}, s::A, p::LightDark1D) where A<:AbstractArray = LightDark1DState(Int64(s[1]), s[2])


# Define some simple policies based on particle belief
mutable struct DummyHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
DummyHeuristic1DPolicy() = DummyHeuristic1DPolicy(0.1)

mutable struct SmartHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
SmartHeuristic1DPolicy() = SmartHeuristic1DPolicy(0.1)

function action(p::DummyHeuristic1DPolicy, b::B) where {B}
    target = 0.0
    μ = mean(b)
    σ = std(b, μ)

    if σ.y < p.thres && -0.5 < μ.y < 0.5
        a = 0
    elseif μ.y < target
        a = 1                   # Right
    elseif μ.y > target
        a = -1                  # Left
    end
    return a
end

function action(p::SmartHeuristic1DPolicy, b::B) where {B}
    μ = mean(b)
    σ = std(b, μ)
    target = 0.0
    if σ.y > p.thres
        target = 5.0
    end
    if σ.y < p.thres && -0.5 < μ.y < 0.5
        a = 0
    elseif μ.y < target
        a = 1                   # Right
    elseif μ.y > target
        a = -1                  # Left
    end
    return a
end

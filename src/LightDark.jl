# A one-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg

# docstring should be improved
"""
    LightDark1D

A one-dimensional light dark problem. The goal is to be near 0. Observations are noisy measurements of the position.

Model
-----

   -3-2-1 0 1 2 3
...| | | | | | | | ...
          G   S

Here G is the goal. S is the starting location

The sigma function should return the standard deviation of the observation distribution.

NaN is used to represent the terminal state
"""
mutable struct LightDark1D{SIGMA<:Function} <: POMDPs.POMDP{Float64,Int,Float64}
    discount_factor::Float64
    correct_r::Float64   # usually positive
    incorrect_r::Float64 # usually negative
    movement_cost::Float64 # positive indicates there is a penalty
    sigma::SIGMA
end

default_sigma(x::Float64) = abs(x - 5)/sqrt(2) + 1e-2
LightDark1D() = LightDark1D(0.9, 10.0, -10.0, 0.0, default_sigma)

discount(p::LightDark1D) = p.discount_factor
actions(::LightDark1D) = -1:1
n_actions(p::LightDark1D) = length(actions(p))
initialstate_distribution(pomdp::LightDark1D) = Normal(2, 3)
isterminal(p::LightDark1D, s::Float64) = isnan(s)

observation(p::LightDark1D, s::Float64, a::Int, sp::Float64) = Normal(s+a, p.sigma(s+a)) # not based on sp to handle NaNs
observation(p::LightDark1D, sp::Float64) = Normal(sp, p.sigma(sp))

function transition(p::LightDark1D, s::Float64, a::Int)
    if a == 0
        return Deterministic(NaN)
    else
        return Deterministic(s + a)
    end
end
        
function reward(p::LightDark1D, s, a)
    if a == 0
        if abs(s) < 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost*a
    end
end

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
    if σ > p.thres
        target = 5.0
    end
    if σ < p.thres && -0.5 < μ < 0.5
        a = 0
    elseif μ < target
        a = 1                   # Right
    elseif μ > target
        a = -1                  # Left
    end
    return a
end

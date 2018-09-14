# A one-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg

import Base: ==, +, *, -

struct LightDark1DState
    status::Int64
    y::Float64
end

==(s1::LightDark1DState, s2::LightDark1DState) = (s1.status == s2.status) && (s1.y == s2.y)
*(n::Number, s::LightDark1DState) = LightDark1DState(s.status, n*s.y)

Base.hash(s::LightDark1DState, h::UInt64=zero(UInt64)) = hash(s.status, hash(s.y, h))
copy(s::LightDark1DState) = LightDark1DState(s.status, s.y)

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
mutable struct LightDark1D <: POMDPs.POMDP{LightDark1DState,Int64,Float64}
    discount_factor::Float64
    correct_r::Float64
    incorrect_r::Float64
    step_size::Float64
    movement_cost::Float64
end

LightDark1D() = LightDark1D(0.9, 10, -10, 1, 0)

discount(p::LightDark1D) = p.discount_factor

isterminal(::LightDark1D, act::Int64) = act == 0

isterminal(::LightDark1D, s::LightDark1DState) = s.status < 0


mutable struct LightDark1DActionSpace
    actions::NTuple{3,Int64}
end
Base.iterate(space::LightDark1DActionSpace, state) = Base.iterate(space.actions, state)
Base.iterate(space::LightDark1DActionSpace) = Base.iterate(space.actions)


Base.length(asp::LightDark1DActionSpace) = length(asp.actions)
actions(::LightDark1D) = LightDark1DActionSpace((-1, 0, 1)) # Left Stop Right
actions(pomdp::LightDark1D, s::LightDark1DState) = actions(pomdp)
dimensions(::LightDark1DActionSpace) = 1
n_actions(p::LightDark1D) = length(actions(p))

rand(rng::AbstractRNG, asp::LightDark1DActionSpace) = asp.actions[rand(rng, 1:3)]

struct LDNormalStateDist
    mean::Float64
    std::Float64
end

Base.eltype(::Type{LDNormalStateDist}) = LightDark1DState

function rand(rng::AbstractRNG, d::LDNormalStateDist)
    return LightDark1DState(0, d.mean + randn(rng)*d.std)
end

function initialstate_distribution(pomdp::LightDark1D)
    return LDNormalStateDist(2, 3)
end

sigma(x::Float64) = abs(x - 5)/sqrt(2) + 1e-2

function observation(p::LightDark1D, s::LightDark1DState, a::Int, sp::LightDark1DState)
    return Normal(sp.y, sigma(sp.y))
end

function generate_o(p::LightDark1D, s::Union{LightDark1DState,Nothing}, a::Union{Int,Nothing}, sp::LightDark1DState, rng::AbstractRNG)
    return sp.y + Base.randn(rng)*sigma(sp.y)
end
generate_o(p::LightDark1D, sp::Union{LightDark1DState,Nothing}, rng::AbstractRNG) = sp.y + Base.randn(rng)*sigma(sp.y)

function generate_s(p::LightDark1D, s::LightDark1DState, a::Int, rng::AbstractRNG)
    if s.status < 0                  # Terminal state
        return s
    end
    if a == 0                   # Enter
        return LightDark1DState(-1, s.y)
    else
        return LightDark1DState(s.status, s.y+a)
    end
end

function generate_sor(p::LightDark1D, s::LightDark1DState, a::Int, rng::AbstractRNG)
    if s.status < 0                  # Terminal state
        sprime = s
        o = generate_o(p, nothing, nothing, sprime, rng)
        r = 0.0                   # Penalty?
        return sprime, o, r
    end
    if a == 0                   # Enter
        sprime = LightDark1DState(-1, s.y)
        if abs(s.y) < 1         # Correct loc is near 0
            r = p.correct_r     # Correct
        else
            r = p.incorrect_r   # Incorrect
        end
    else
        sprime = LightDark1DState(s.status, s.y+a)
        r = 0.0
    end
    o = generate_o(p, s, a, sprime, rng)
    return sprime, o, r
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
        return 0.0
    end
end


convert_s(::Type{A}, s::LightDark1DState, p::LightDark1D) where A<:AbstractArray = eltype(A)[s.status, s.y]
convert_s(::Type{LightDark1DState}, s::A, p::LightDark1D) where A<:AbstractArray = LightDark1DState(Int64(s[1]), s[2])

# XXX this is specifically for MCVI
# it is also implemented in the MCVI tests
function init_lower_action(p::LightDark1D)
    return 0 # Worst? This depends on the initial state? XXX
end

observation(p::LightDark1D, sp::LightDark1DState) = Normal(sp.y, sigma(sp.y))

#=
gauss(s::Float64, x::Float64) = 1 / sqrt(2*pi) / s * exp(-1*x^2/(2*s^2))
function obs_weight(p::LightDark1D, s::LightDark1DState, obs::Float64)
    return gauss(sigma(s.y), s.y-obs)
end

# old - this should not be there
function pdf(s::LightDark1DState, obs::Float64)
    return gauss(sigma(s.y), s.y-obs)
end
=#

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

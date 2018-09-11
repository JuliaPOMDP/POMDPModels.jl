# A one-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg


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
"""
mutable struct LightDark1D{SIGMA<:Function} <: POMDPs.POMDP{Union{Float64,TerminalState},Int,Float64}
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

observation(p::LightDark1D, s::Float64, a::Int, sp) = Normal(s+a, p.sigma(s+a))
observation(p::LightDark1D, sp::Float64) = Normal(sp, p.sigma(sp))

function transition(p::LightDark1D, s::Float64, a::Int)
    if a == 0
        return Deterministic(terminalstate)
    else
        return Deterministic(s + a)
    end
end
        
# function generate_s(p::LightDark1D, s::LightDark1DState, a::Int, rng::AbstractRNG)
#     if s.status < 0                  # Terminal state
#         return s
#     end
#     if a == 0                   # Enter
#         return LightDark1DState(-1, s.y)
#     else
#         return LightDark1DState(s.status, s.y+a)
#     end
# end

function reward(p::LightDark1D, s::Float64, a::Int)
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


convert_s(::Type{A}, s::TerminalState, p::LightDark1D) where A<:AbstractArray = convert(A, [NaN])
function convert_s(::Type{Union{Float64,TerminalState}}, s::A, p::LightDark1D) where A<:AbstractArray
    if isnan(first(s))
        return terminalstate
    else
        return convert(Float64, first(s))
    end
end

# function generate_sor(p::LightDark1D, s::LightDark1DState, a::Int, rng::AbstractRNG)
#     if s.status < 0                  # Terminal state
#         sprime = s
#         o = generate_o(p, nothing, nothing, sprime, rng)
#         r = 0.0                   # Penalty?
#         return sprime, o, r
#     end
#     if a == 0                   # Enter
#         sprime = LightDark1DState(-1, s.y)
#         if abs(s.y) < 1         # Correct loc is near 0
#             r = p.correct_r     # Correct
#         else
#             r = p.incorrect_r   # Incorrect
#         end
#     else
#         sprime = LightDark1DState(s.status, s.y+a)
#         r = 0.0
#     end
#     o = generate_o(p, s, a, sprime, rng)
#     return sprime, o, r
# end


# XXX this is specifically for MCVI
# it is also implemented in the MCVI tests
# function init_lower_action(p::LightDark1D)
#     return 0 # Worst? This depends on the initial state? XXX
# end


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
# function generate_o(p::LightDark1D, s::Union{LightDark1DState,Nothing}, a::Union{Int,Nothing}, sp::LightDark1DState, rng::AbstractRNG)
#     return sp.y + Base.randn(rng)*sigma(sp.y)
# end
# generate_o(p::LightDark1D, sp::Union{LightDark1DState,Nothing}, rng::AbstractRNG) = sp.y + Base.randn(rng)*sigma(sp.y)


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

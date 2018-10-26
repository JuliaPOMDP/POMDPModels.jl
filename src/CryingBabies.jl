# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

mutable struct BabyPOMDP <: POMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
end
BabyPOMDP(r_feed, r_hungry) = BabyPOMDP(r_feed, r_hungry, 0.1, 0.8, 0.1, 0.9)
BabyPOMDP() = BabyPOMDP(-5., -10.)

updater(problem::BabyPOMDP) = DiscreteUpdater(problem)

actions(::BabyPOMDP) = (true, false)
actionindex(::BabyPOMDP, a::Bool) = a + 1
n_actions(::BabyPOMDP) = 2
states(::BabyPOMDP) = (true, false)
stateindex(::BabyPOMDP, s::Bool) = s + 1
n_states(::BabyPOMDP) = 2
observations(::BabyPOMDP) = (true, false)
obsindex(::BabyPOMDP, o::Bool) = o + 1
n_observations(::BabyPOMDP) = 2

# start knowing baby is not not hungry
initialstate_distribution(::BabyPOMDP) = BoolDistribution(0.0)

function transition(pomdp::BabyPOMDP, s::Bool, a::Bool)
    if a # fed
        return BoolDistribution(0.0)
    elseif s # did not feed when hungry
        return BoolDistribution(1.0)
    else # did not feed when not hungry
        return BoolDistribution(pomdp.p_become_hungry)
    end
end

function observation(pomdp::BabyPOMDP, a::Bool, sp::Bool)
    if sp # hungry
        return BoolDistribution(pomdp.p_cry_when_hungry)
    else
        return BoolDistribution(pomdp.p_cry_when_not_hungry)
    end
end

function reward(pomdp::BabyPOMDP, s::Bool, a::Bool)
    r = 0.0
    if s # hungry
        r += pomdp.r_hungry
    end
    if a # feed
        r += pomdp.r_feed
    end
    return r
end


discount(p::BabyPOMDP) = p.discount

function generate_o(p::BabyPOMDP, s::Bool, rng::AbstractRNG)
    d = observation(p, true, s) # obs distrubtion not action dependant
    return rand(rng, d)
end

# some example policies
mutable struct Starve <: Policy end
action(::Starve, ::B) where {B} = false
updater(::Starve) = NothingUpdater()

mutable struct AlwaysFeed <: Policy end
action(::AlwaysFeed, ::B) where {B} = true
updater(::AlwaysFeed) = NothingUpdater()

# feed when the previous observation was crying - this is nearly optimal
mutable struct FeedWhenCrying <: Policy end
updater(::FeedWhenCrying) = PreviousObservationUpdater()
function action(::FeedWhenCrying, b::Union{Nothing, Bool})
    if b == nothing || b == false # not crying (or null)
        return false
    else # is crying
        return true
    end
end
action(::FeedWhenCrying, b::Bool) = b
action(p::FeedWhenCrying, b::Missing) = false
# assume the second argument is a distribution
action(::FeedWhenCrying, d::Any) = pdf(d, true) > 0.5

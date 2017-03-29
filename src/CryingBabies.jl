# Crying baby problem described in DMU book
# State: hungry = true; not hungry = false
# Action: feed = true; do nothing = false
# Observation: crying = true; not crying = false

type BabyPOMDP <: POMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64
end
BabyPOMDP(r_feed, r_hungry) = BabyPOMDP(r_feed, r_hungry, 0.1, 0.8, 0.1, 0.9)
BabyPOMDP() = BabyPOMDP(-5., -10.)

# TODO: this should be moved to POMDPDistributions.jl
immutable BoolDistribution
    p::Float64 # probability of true
end
BoolDistribution() = BoolDistribution(0.0)
pdf(d::BoolDistribution, s::Bool) = s ? d.p : 1.0-d.p
iterator(d::BoolDistribution) = [true, false]
Base.length(d::BoolDistribution) = 2
index(d::BoolDistribution, s::Bool) = s ? 1:2
Base.convert(t::Type{DiscreteBelief}, b::BoolDistribution) = DiscreteBelief([b.p, 1.0-b.p])

type BabyBeliefUpdater <: Updater{BoolDistribution}
    problem::BabyPOMDP
end
updater(problem::BabyPOMDP) = BabyBeliefUpdater(problem)

initial_state_distribution(::BabyPOMDP) = BoolDistribution(0.0)

n_states(::BabyPOMDP) = 2
state_index(::BabyPOMDP, s::Bool) = s ? 1 : 2
action_index(::BabyPOMDP, s::Bool) = s ? 1 : 2
n_actions(::BabyPOMDP) = 2
n_observations(::BabyPOMDP) = 2

function transition(pomdp::BabyPOMDP, s::Bool, a::Bool)
    if !a && s # did not feed when hungry
        return BoolDistribution(1.0)
    elseif a # fed
        return BoolDistribution(0.0)
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
observation(pomdp::BabyPOMDP, s::Bool, a::Bool, sp::Bool) = observation(pomdp, a, sp)

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

rand(rng::AbstractRNG, d::BoolDistribution) = rand(rng) <= d.p

function update(bu::BabyBeliefUpdater, old::BoolDistribution, a::Bool, o::Bool)
    p = bu.problem
    if a # feed
        return BoolDistribution(0.0)
    else # did not feed
        ph = old.p + (1.0-old.p)*p.p_become_hungry # this is from the system dynamics
        # bayes rule
        if o # crying
            ph = (p.p_cry_when_hungry*ph)/(p.p_cry_when_hungry*ph + p.p_cry_when_not_hungry*(1.0-ph))
        else # not crying
            ph = ((1.0-p.p_cry_when_hungry)*ph)/((1.0-p.p_cry_when_hungry)*ph + (1.0-p.p_cry_when_not_hungry)*(1.0-ph))
        end
        return BoolDistribution(ph)
    end
end

dimensions(::BoolDistribution) = 1

discount(p::BabyPOMDP) = p.discount

function generate_o(p::BabyPOMDP, s::Bool, rng::AbstractRNG)
    d = observation(p, true, s) # obs distrubtion not action dependant
    return rand(rng, d)
end

# same for both state and observation
vec(p::BabyPOMDP, so::Bool) = Float64[so]

# some example policies
type Starve <: Policy end
action{B}(::Starve, ::B) = false
updater(::Starve) = VoidUpdater()

type AlwaysFeed <: Policy end
action{B}(::AlwaysFeed, ::B) = true
updater(::AlwaysFeed) = VoidUpdater()

# feed when the previous observation was crying - this is nearly optimal
type FeedWhenCrying <: Policy end
updater(::FeedWhenCrying) = PreviousObservationUpdater{Bool}()
function action(::FeedWhenCrying, b::Nullable{Bool})
    if get(b, false) == false # not crying (or null)
        return false
    else # is crying
        return true
    end
end
action(::FeedWhenCrying, b::Bool) = b

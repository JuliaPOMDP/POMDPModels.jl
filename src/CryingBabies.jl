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

type BabyStateDistribution <: Belief{Bool}
    p_hungry::Float64 # probability of being hungry
end
BabyStateDistribution() = BabyStateDistribution(0.0)

type BabyObservationDistribution <: AbstractDistribution{Bool}
    p_crying::Float64 # probability of crying
end
BabyObservationDistribution() = BabyObservationDistribution(0.0)

type BabyBeliefUpdater <: BeliefUpdater{Bool, Bool, Bool}
    problem::BabyPOMDP
end
updater(problem::BabyPOMDP) = BabyBeliefUpdater(problem)

create_transition_distribution(::BabyPOMDP) = BabyStateDistribution()
create_observation_distribution(::BabyPOMDP) = BabyObservationDistribution()
create_belief(::BabyBeliefUpdater) = BabyStateDistribution()
create_belief(::BabyPOMDP) = BabyStateDistribution()
initial_belief(::BabyPOMDP) = BabyStateDistribution(0.0)

n_states(::BabyPOMDP) = 2
n_actions(::BabyPOMDP) = 2
n_observations(::BabyPOMDP) = 2

function transition(pomdp::BabyPOMDP, s::Bool, a::Bool, d::BabyStateDistribution=BabyStateDistribution())
    if !a && s # don't feed when hungry
        d.p_hungry = 1.0
    elseif a # feed
        d.p_hungry = 0.0
    else # don't feed when not hungry
        d.p_hungry = pomdp.p_become_hungry
    end
    return d
end

function observation(pomdp::BabyPOMDP, s::Bool, a::Bool, sp::Bool, d::BabyObservationDistribution=create_observation_distribution(pomdp))
    if sp # hungry
        d.p_crying = pomdp.p_cry_when_hungry
    else
        d.p_crying = pomdp.p_cry_when_not_hungry
    end
    return d
end

function reward(pomdp::BabyPOMDP, s::Bool, a::Bool, sp::Bool)
    r = 0.0
    if s # hungry
        r += pomdp.r_hungry
    end
    if a # feed
        r += pomdp.r_feed
    end
    return r
end

rand(rng::AbstractRNG, d::BabyStateDistribution, s::Bool=false) = (rand(rng) <= d.p_hungry)
rand(rng::AbstractRNG, d::BabyObservationDistribution, o::Bool=false) = (rand(rng) <= d.p_crying)

function update(bu::BabyBeliefUpdater, old::BabyStateDistribution, a::Bool, o::Bool, b::BabyStateDistribution=BabyStateDistribution())
    p = bu.problem
    if a # feed
        b.p_hungry = 0.0
    else # did not feed
        b.p_hungry = old.p_hungry + (1.0-old.p_hungry)*p.p_become_hungry # this is from the system dynamics
        # bayes rule
        if o # crying
            b.p_hungry = (p.p_cry_when_hungry*b.p_hungry)/(p.p_cry_when_hungry*b.p_hungry + p.p_cry_when_not_hungry*(1.0-b.p_hungry))
        else # not crying
            b.p_hungry = ((1.0-p.p_cry_when_hungry)*b.p_hungry)/((1.0-p.p_cry_when_hungry)*b.p_hungry + (1.0-p.p_cry_when_not_hungry)*(1.0-b.p_hungry))
        end
    end
    return b
end

dimensions(::BabyObservationDistribution) = 1
dimensions(::BabyStateDistribution) = 1

type BoolSpace <: AbstractSpace{Bool} end
iterator(bs::BoolSpace) = bs
Base.start(::BoolSpace) = 0
Base.done(::BoolSpace, st::Int) = st > 1
Base.next(::BoolSpace, st::Int) = (st==0, st+1)

states(::BabyPOMDP) = BoolSpace()
actions(::BabyPOMDP, s::Bool=true, as::BoolSpace=BoolSpace()) = as

discount(p::BabyPOMDP) = p.discount
# isterminal(::BabyPOMDP, ::BabyState) = false

# some example policies
type Starve <: Policy end
action(::Starve, ::Belief, a=false) = false
updater(::Starve) = EmptyUpdater()

type AlwaysFeed <: Policy end
action(::AlwaysFeed, ::Belief, a=true) = true
updater(::AlwaysFeed) = EmptyUpdater()

# feed when the previous observation was crying - this is nearly optimal
#=
type FeedWhenCrying <: Policy end
updater(::FeedWhenCrying) = PreviousObservationUpdater()
function action(::FeedWhenCrying, b::PreviousObservation, a=false)
    if b.observation == nothing || b.observation == false # not crying
        return false
    else # is crying
        return true
    end
end
# action(::FeedWhenCrying, b::BabyStateDistribution, a=false) = false
=#

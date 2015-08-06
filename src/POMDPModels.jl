module POMDPModels

using POMDPs
# using Distributions # this takes a while to import. For various reasons, I (Zach) think we should avoid using Distributions.jl
using POMDPToolbox

#= # it appears that you have to import these all by name
import POMDPs: n_states, n_actions, n_observations
import POMDPs: states, states!, actions, actions!, observations, observations!
import POMDPs: create_state, create_observation
import POMDPs: create_transition_distribution, create_observation_distribution, transition!, reward
import POMDPs: rand, rand!, length, index, weight, domain, isterminal
=#

# XXX this is not a complete list - it's only the ones needed to test simulate
import POMDPs.create_state
import POMDPs.create_observation
import POMDPs.create_transition_distribution
import POMDPs.create_observation_distribution
import POMDPs.action
import POMDPs.isterminal
import POMDPs.reward
import POMDPs.transition!
import POMDPs.observation!
import POMDPs.update_belief!
import POMDPs.discount
import POMDPs.actions
import Base.rand! # hmmm... is this right?
import Base.rand

export
    # Grid World
    GridWorld,
    GridWorldState,
    GridWorldAction,
    GridWorldDistribution,
    # CryingBabies
    BabyPOMDP,
    BabyState,
    BabyObservation,
    BabyAction,
    BabyStateDistribution,
    BabyObservationDistribution,
    Starve,
    AlwaysFeed,
    FeedWhenCrying,
    # Commons
    n_states,
    n_actions,
    n_observations,
    states,
    states!,
    actions,
    actions!,
    observations,
    observations!,
    create_action,
    create_state,
    create_observation,
    create_transition,
    reward,
    transition!,
    length,
    index,
    weight,
    domain,
    rand,
    rand!,
    isterminal

include("GridWorlds.jl")
include("CryingBabies.jl")
#include("TigerPOMDPs.jl")

end # module

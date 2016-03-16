#################################################################
# This module provides implementations of MDP and POMDP problems
# in the POMDPs.jl format. These implementations serve as 
# examples for new users and as benchmark problems for testing
# and valdiation.
#################################################################

module POMDPModels

using POMDPs
using POMDPToolbox

import POMDPs: n_states, n_actions, n_observations # space sizes for discrete problems
import POMDPs: discount, states, actions, observations # model functions
import POMDPs: transition, observation, reward, isterminal # model functions
import POMDPs: create_state, create_action, create_observation # s,a,o initialization
import POMDPs: length, index, weight # discrete distribution functions
import POMDPs: rand, pdf # common distribution functions
import POMDPs: iterator, dimensions # space functions
import POMDPs: create_transition_distribution, create_observation_distribution, create_belief, initial_belief 
import POMDPs: update, updater
import POMDPs: upperbound

# for example policies
import POMDPs: Policy, create_policy, action

import Base.rand! 
import Base.rand
import Base.==
import Base.hash

export
    # Tiger
    TigerPOMDP,
    TigerState,
    TigerAction,
    TigerObservation,
    AbstractTigerDistribution,
    TigerStateDistribution,
    TigerObservationDistribution,
    TigerStateSpace,
    TigerActionSpace,
    TigerObservationSpace,
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
    BabyBeliefUpdater,
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
    observation,
    create_action,
    create_state,
    create_observation,
    create_observation_distribution,
    create_transition_distribution,
    create_belief,
    reward,
    transition,
    length,
    index,
    weight,
    #domain,
    iterator,
    rand,
    rand!,
    isterminal,
    discount,
    initial_belief,
    create_belief,
    belief,
    update_belief!,
    pdf,
    dimensions,
    upperbound,
    getindex

# commenting out GridWorlds and Tiger so that I can focus on CryingBabies initially
# include("GridWorlds.jl")
include("CryingBabies.jl")
# include("TigerPOMDPs.jl")

end # module

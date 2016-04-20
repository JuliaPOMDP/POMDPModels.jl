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
import POMDPs: state_index, action_index, obs_index
import POMDPs: discount, states, actions, observations # model functions
import POMDPs: transition, observation, reward, isterminal # model functions
import POMDPs: length, index, weight # discrete distribution functions
import POMDPs: rand, pdf # common distribution functions
import POMDPs: iterator, dimensions # space functions
import POMDPs: create_state, create_action, create_observation
import POMDPs: create_transition_distribution, create_observation_distribution, create_belief, initial_state_distribution
import POMDPs: update, updater
import POMDPs: upperbound

# for example policies
import POMDPs: Policy, create_policy, action

import Base.rand! 
import Base.rand
import Base.==
import Base.hash

import GenerativeModels: initial_state

export
    # Tiger
    TigerPOMDP,
    TigerDistribution,
    TigerStateSpace,
    TigerActionSpace,
    TigerObservationSpace,
    TigerBeliefUpdater,
    # Grid World
    GridWorld,
    GridWorldState,
    GridWorldAction,
    GridWorldDistribution,
    # CryingBabies
    BabyPOMDP,
    BoolDistribution,
    BabyBeliefUpdater,
    Starve,
    AlwaysFeed,
    FeedWhenCrying,
    # Commons
    n_states,
    n_actions,
    n_observations,
    state_index,
    action_index,
    observation_index,
    states,
    actions,
    observations,
    observation,
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
    isterminal,
    discount,
    initial_belief,
    create_belief,
    belief,
    update,
    pdf,
    dimensions,
    upperbound,
    getindex

include("GridWorlds.jl")
include("CryingBabies.jl")
include("TigerPOMDPs.jl")

end # module

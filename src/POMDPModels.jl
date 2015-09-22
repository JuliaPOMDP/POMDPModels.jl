#################################################################
# This module provides implementations of MDP and POMDP problems
# in the POMDPs.jl format. These implementations serve as 
# examples for new users and as benchmark problems for testing
# and valdiation.
#################################################################

module POMDPModels

using POMDPs
using POMDPToolbox

<<<<<<< Updated upstream
import POMDPs: domain, states, actions, actions, observations, observation
import POMDPs: create_transition_distribution, create_observation_distribution, create_belief
import POMDPs: create_state, create_observation
import POMDPs: reward, transition, observation
import POMDPs: n_states, n_actions, n_observations
import POMDPs: length, weight, index
import POMDPs: belief
import POMDPs: discount
import POMDPs: action, create_action
import POMDPs: pdf

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
    #GridWorld,
    #GridWorldState,
    #GridWorldAction,
    #GridWorldDistribution,
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
    domain,
    rand,
    rand!,
    isterminal,
    discount,
    initial_belief,
    create_belief,
    belief,
    update_belief!,
    pdf

#include("GridWorlds.jl")
include("CryingBabies.jl")
include("TigerPOMDPs.jl")

end # module

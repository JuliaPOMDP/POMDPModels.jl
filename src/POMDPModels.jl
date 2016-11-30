__precompile__()

#################################################################
# This module provides implementations of MDP and POMDP problems
# in the POMDPs.jl format. These implementations serve as
# examples for new users and as benchmark problems for testing
# and valdiation.
#################################################################

module POMDPModels

using POMDPs
using POMDPToolbox
using Distributions

import POMDPs: n_states, n_actions, n_observations # space sizes for discrete problems
import POMDPs: state_index, action_index, obs_index
import POMDPs: discount, states, actions, observations # model functions
import POMDPs: transition, observation, reward, isterminal # model functions
import POMDPs: rand, pdf # common distribution functions
import POMDPs: iterator, dimensions # space functions
import POMDPs: create_state, create_action, create_observation
import POMDPs: create_transition_distribution, create_observation_distribution, create_belief, initial_state_distribution
import POMDPs: update, updater
import POMDPs: vec

# for example policies
import POMDPs: Policy, create_policy, action

import Base.rand!
import Base.rand
import Base.==
import Base.hash

import GenerativeModels: initial_state, generate_s, generate_o

# for grid world visualization
using TikzPictures

include("TigerPOMDPs.jl")
export
    TigerPOMDP,
    TigerDistribution,
    TigerStateSpace,
    TigerActionSpace,
    TigerObservationSpace,
    TigerBeliefUpdater


include("GridWorlds.jl")
export
    GridWorld,
    GridWorldState,
    GridWorldAction,
    GridWorldDistribution,
    static_reward,
    plot

include("CryingBabies.jl")
export
    BabyPOMDP,
    BoolDistribution,
    BabyBeliefUpdater,
    Starve,
    AlwaysFeed,
    FeedWhenCrying

include("MountainCar.jl")
export
    MountainCar,
    Energize

include("InvertedPendulum.jl")
export
    InvertedPendulum

include("Discrete.jl")
export
    DiscreteMDP,
    DiscretePOMDP

include("Random.jl")
export
    RandomMDP,
    RandomPOMDP

include("TMazes.jl")
export
    TMaze,
    TMazeState,
    TMazeSpace,
    TMazeStateSpace,
    MazeOptimal

include("LightDark.jl")
export
    LightDark1D,
    LightDark1DState,
    LightDark1DActionSpace,
    LightDark1DLowerBound,
    LightDark1DUpperBound,
    DummyHeuristic1DPolicy,
    SmartHeuristic1DPolicy

export
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
    create_state,
    create_action,
    create_observation,
    create_belief,
    reward,
    transition,
    length,
    index,
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
    getindex,
    initial_state_distribution,
    vec,
    # generative model
    generate_s,
    generate_o,
    initial_state

end # module

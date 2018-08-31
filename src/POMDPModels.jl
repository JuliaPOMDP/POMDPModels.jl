#################################################################
# This module provides implementations of MDP and POMDP problems
# in the POMDPs.jl format. These implementations serve as
# examples for new users and as benchmark problems for testing
# and valdiation.
#################################################################

module POMDPModels

using POMDPs
using POMDPModelTools
using BeliefUpdaters
using Distributions
using StaticArrays
using AutoHashEquals
using StatsBase
using Random
using Printf
using Parameters

using POMDPs

import Base: ==, hash
import Random: rand, rand!
import Distributions: pdf

import POMDPs: initialstate, generate_s, generate_o, generate_sor, support, discount, isterminal
import POMDPs: actions, n_actions, action_index, action
import POMDPs: states, n_states, state_index, transition
import POMDPs: observations, observation, n_observations, obs_index
import POMDPs: initialstate, initialstate_distribution
import POMDPs: updater, update
import POMDPs: reward
import POMDPs: convert_s, convert_a, convert_o

# # for grid world visualization
# using TikzPictures

include("TigerPOMDPs.jl")
export
    TigerPOMDP,
    TigerDistribution,
    TigerStateSpace,
    TigerActionSpace,
    TigerObservationSpace,
    TigerBeliefUpdater,
    TIGER_LEFT,
    TIGER_RIGHT,
    TIGER_LISTEN,
    TIGER_OPEN_LEFT,
    TIGER_OPEN_RIGHT


include("GridWorlds.jl")
export
    GridWorld,
    GridWorldState,
    GridWorldAction,
    GridWorldActionSpace,
    GridWorldStateSpace,
    GridWorldDistribution,
    static_reward,
    plot

include("CryingBabies.jl")
export
    BabyPOMDP,
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

include("Tabular.jl")
export
    TabularMDP,
    TabularPOMDP

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
    DummyHeuristic1DPolicy,
    SmartHeuristic1DPolicy

export
    n_states,
    n_actions,
    n_observations,
    state_index,
    action_index,
    obs_index,
    states,
    actions,
    observations,
    observation,
    reward,
    transition,
    length,
    index,
    #domain,
    support,
    rand,
    isterminal,
    discount,
    initial_belief,
    belief,
    update,
    pdf,
    dimensions,
    upperbound,
    getindex,
    initialstate_distribution,
    vec,
    # generative model
    generate_s,
    generate_o,
    generate_sor,
    initialstate

end # module

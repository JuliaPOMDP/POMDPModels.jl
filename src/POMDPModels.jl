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
using StaticArrays
using AutoHashEquals
using StatsBase

importall POMDPs

import Base.rand!
import Base.rand
import Base.==
import Base.hash

import POMDPs: initial_state, generate_s, generate_o, generate_sor

# for grid world visualization
using TikzPictures

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
    iterator,
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
    initial_state_distribution,
    vec,
    # generative model
    generate_s,
    generate_o,
    generate_sor,
    initial_state

end # module

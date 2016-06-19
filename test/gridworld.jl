# Install related packages on first run
# using POMDPs
# POMDPs.add("POMDPModels")
# POMDPs.add("POMDPToolbox")
# POMDPs.add("GenerativeModels")

using POMDPModels
using POMDPToolbox
using Base.Test

problem = GridWorld()

policy = RandomPolicy(problem)

sim = RolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, GridWorldState(1,1))

sv = vec(problem, GridWorldState(1, 1, false, false))
@test sv == [1.0, 1.0, 0.0]
sv = vec(problem, GridWorldState(5, 3, true, false))
@test sv == [5.0, 3.0, 1.0]

trans_prob_consistancy_check(problem)

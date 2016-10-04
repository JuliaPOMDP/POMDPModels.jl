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

sv = vec(problem, GridWorldState(1, 1, false))
@test sv == [1.0, 1.0]
sv = vec(problem, GridWorldState(5, 3, false))
@test sv == [5.0, 3.0]

@test GridWorldState(1,1,false) == GridWorldState(1,1,false)
@test hash(GridWorldState(1,1,false)) == hash(GridWorldState(1,1,false))
@test GridWorldState(1,2,false) != GridWorldState(1,1,false)
@test GridWorldState(1,2,true) == GridWorldState(1,1,true)
@test hash(GridWorldState(1,2,true)) == hash(GridWorldState(1,1,true))

trans_prob_consistancy_check(problem)

plot(problem, state=GridWorldState(1,1))

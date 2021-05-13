using POMDPModels
using Test
using POMDPTesting

pomdp = MiniHallway()

@test has_consistent_distributions(pomdp)

@test simulate(RolloutSimulator(max_steps=100), pomdp, RandomPolicy(pomdp)) >= 0.0

@test stateindex(pomdp, 5) == 5
@test actionindex(pomdp, 1) == 1
@test obsindex(pomdp, 1) == 1

@test reward(pomdp, 11, 1) == 1

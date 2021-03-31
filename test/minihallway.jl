using POMDPModels
using Test
using POMDPTesting

pomdp = MiniHallway()

@test has_consistent_distributions(pomdp)

@test simulate(RolloutSimulator(max_steps=100), pomdp, RandomPolicy(pomdp)) >= 0.0

using POMDPModels
using Test
using POMDPTesting

pomdp = MiniHallway()

@test has_consistent_distributions(pomdp)

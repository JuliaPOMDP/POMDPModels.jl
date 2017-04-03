using POMDPModels
using POMDPToolbox
using Base.Test


problem = InvertedPendulum()
policy = RandomPolicy(problem)
sim = RolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, initial_state(problem, MersenneTwister(2)))

sv = convert(problem, (0.5, 0.25))
@test sv == [0.5, 0.25]
s = convert(problem, sv)
@test s == (0.5, 0.25)

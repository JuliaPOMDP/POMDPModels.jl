using POMDPs
using POMDPModels
using POMDPToolbox

problem = GridWorld()

policy = RandomPolicy(problem)

sim = MDPRolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, GridWorldState(1,1))

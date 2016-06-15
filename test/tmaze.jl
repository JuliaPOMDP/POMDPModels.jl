using Base.Test

using POMDPModels
using POMDPToolbox
using POMDPs

problem = TMaze(10)

policy = RandomPolicy(problem, rng=MersenneTwister(2))
sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
        
simulate(sim, problem, policy, updater(policy), initial_state_distribution(problem))

probability_check(problem)

using Base.Test

using POMDPModels
using POMDPToolbox
using POMDPs

problem = BabyPOMDP()

# starve policy
# when the baby is never fed, the reward for starting in the hungry state should be -100
sim = RolloutSimulator()
sim.eps = 0.0001
sim.initial_state = true
ib = nothing
policy = Starve()
r = simulate(sim, problem, policy, updater(policy), ib)
@test_approx_eq_eps r -100.0 0.01

probability_check(problem)

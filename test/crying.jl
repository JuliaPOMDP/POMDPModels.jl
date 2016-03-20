# test simple policies on the crying baby problem

using Base.Test

using POMDPModels
using POMDPToolbox
using POMDPs

problem = BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.9)

# starve policy
# when the baby is never fed, the reward for starting in the hungry state should be -100
sim = RolloutSimulator()
sim.eps = 0.0001
sim.initial_state = true
ib = EmptyBelief()
policy = Starve()
r = simulate(sim, problem, policy, updater(policy), ib)
@test_approx_eq_eps r -100.0 0.01

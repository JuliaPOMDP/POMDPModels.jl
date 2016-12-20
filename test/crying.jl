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

# test generate_o
o = generate_o(problem, true, MersenneTwister(1))
@test o == 1
# test vec
ov = vec(problem, true)
@test ov == [1.]

probability_check(problem)

bp =  update(BabyBeliefUpdater(problem),
             BoolDistribution(0.0),
             false,
             true)

@test_approx_eq_eps bp.p 0.47058823529411764 0.0001

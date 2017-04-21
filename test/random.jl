using Base.Test
using POMDPModels
using POMDPToolbox

ns = 10
na = 2
no = 4
disc = 0.95

mdp = RandomMDP(ns, na, disc, rng=MersenneTwister(1))
policy = RandomPolicy(mdp, rng=MersenneTwister(2))
sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
        
simulate(sim, mdp, policy, 1)
trans_prob_consistency_check(mdp)

pomdp = RandomPOMDP(ns, na, no, disc, rng=MersenneTwister(1))
policy = RandomPolicy(pomdp, rng=MersenneTwister(2))
sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
        
simulate(sim, pomdp, policy, updater(policy), initial_state_distribution(pomdp))
probability_check(pomdp)

ov = convert(Array{Float64}, 1, pomdp)
@test ov == [1.]
o = convert(Int, ov, pomdp)
@test o == 1

using Test
using POMDPModels
using Random
# using POMDPSimulators
# using POMDPPolicies
using POMDPs
using POMDPModelTools
using POMDPTesting

let
    ns = 10
    na = 2
    no = 4
    disc = 0.95

    mdp = RandomMDP(ns, na, disc, rng=MersenneTwister(1))
    policy = RandomPolicy(mdp, rng=MersenneTwister(2))
    sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
    
    simulate(sim, mdp, policy, 1)
    # XXX uncomment when POMDPTesting is registered # trans_prob_consistency_check(mdp)

    pomdp = RandomPOMDP(ns, na, no, disc, rng=MersenneTwister(1))
    policy = RandomPolicy(pomdp, rng=MersenneTwister(2))
    sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
    
    simulate(sim, pomdp, policy, updater(policy), initialstate_distribution(pomdp))
    POMDPTesting.probability_check(pomdp)

    ov = convert_o(Array{Float64}, 1, pomdp)
    @test ov == [1.]
    o = convert_o(Int, ov, pomdp)
    @test o == 1

    # to catch anything in the default constructors
    RandomPOMDP()
    RandomMDP()
end

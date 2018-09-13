using POMDPs
using POMDPPolicies
using POMDPSimulators
using POMDPModels
using BenchmarkTools
using Random

mdps = [LegacyGridWorld(terminals = Set()),
        SimpleGridWorld(terminate_in = Set()),
       ]

for m in mdps
    @show typeof(m)
    policy = RandomPolicy(m, rng=MersenneTwister(7))
    rosim = RolloutSimulator(max_steps=10_000, rng=MersenneTwister(2))
    @btime simulate($rosim, $m, $policy)
end

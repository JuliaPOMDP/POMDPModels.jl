# Install related packages on first run
# Pkg.clone("https://github.com/JuliaPOMDP/POMDPModels.jl")
# Pkg.clone("https://github.com/JuliaPOMDP/POMDPToolbox.jl")
# Pkg.clone("https://github.com/JuliaPOMDP/GenerativeModels.jl")

using POMDPs
using POMDPModels
using POMDPToolbox

problem = GridWorld()

policy = RandomPolicy(problem)

sim = RolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, GridWorldState(1,1))

trans_prob_consistancy_check(problem)

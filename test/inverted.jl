using POMDPModels
using POMDPToolbox
using Base.Test


problem = InvertedPendulum()
policy = RandomPolicy(problem)
sim = RolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, initial_state(problem, MersenneTwister(2)))

sv = convert(Array{Float64}, (0.5, 0.25), problem)
@test sv == [0.5, 0.25]
s = convert(Tuple{Float64,Float64}, sv, problem)
@test s == (0.5, 0.25)

using POMDPModels
using Base.Test


problem = InvertedPendulum()
policy = RandomPolicy(problem)
sim = RolloutSimulator(MersenneTwister(1))

simulate(sim, problem, policy, initialstate(problem, MersenneTwister(2)))

sv = convert_s(Array{Float64}, (0.5, 0.25), problem)
@test sv == [0.5, 0.25]
s = convert_s(Tuple{Float64,Float64}, sv, problem)
@test s == (0.5, 0.25)

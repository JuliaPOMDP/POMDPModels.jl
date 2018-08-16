using POMDPModels
using Test

let
    problem = MountainCar()
    policy = RandomPolicy(problem)
    sim = RolloutSimulator(rng=MersenneTwister(1), max_steps=1000)

    r = simulate(sim, problem, policy, initialstate(problem, MersenneTwister(2)))
    @test r < 0.0

    sv = convert_s(Array{Float64}, (0.5, 0.25), problem)
    @test sv == [0.5, 0.25]
    s = convert_s(Tuple{Float64,Float64}, sv, problem)
    @test s == (0.5, 0.25)

    problem = MountainCar(discount=1.0, cost=-0.1, jackpot=100.0)
    policy = RandomPolicy(problem)
    r = simulate(sim, problem, policy, initialstate(problem, MersenneTwister(2)))
    @test r < 0.0
    policy = Energize()
    r = simulate(sim, problem, policy, initialstate(problem, MersenneTwister(2)))
    @test r > 0.0
end

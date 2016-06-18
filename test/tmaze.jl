using Base.Test
using POMDPModels
using POMDPToolbox
using POMDPs

problem = TMaze(10)

policy = RandomPolicy(problem, rng=MersenneTwister(2))
sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
        
simulate(sim, problem, policy, updater(policy), initial_state_distribution(problem))

probability_check(problem)

function test_obs(s::TMazeState, o::Int64)
    ot = generate_o(problem, s, MersenneTwister(1))
    @test ot == o 
end

test_obs(TMazeState(1, :north, false), 1) # north sign
test_obs(TMazeState(1, :south, false), 2) # south sign
test_obs(TMazeState(5, :south, false), 3) # corridor
test_obs(TMazeState(11, :south, false), 4) # junction
test_obs(TMazeState(11, :south, true), 5) # terminal

ov = vec(problem, 1)
@test ov == [1.]

sv = vec(problem, TMazeState(1, :north, false))
@test sv == [1.0, 0.0]
sv = vec(problem, TMazeState(1, :south, false))
@test sv == [1.0, 1.0]


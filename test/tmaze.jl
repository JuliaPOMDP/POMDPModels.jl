using Test
using POMDPModels
using POMDPs
using POMDPTesting

problem = TMaze(n=10)

policy = RandomPolicy(problem, rng=MersenneTwister(2))
sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)

simulate(sim, problem, policy, updater(policy), initialstate_distribution(problem))

POMDPTesting.probability_check(problem)

function test_obs(s, o)
    ot = gen(DDNNode{:o}(), problem, s, MersenneTwister(1))
    @test ot == o
end

test_obs(TMazeState(1, :north), 1) # north sign
test_obs(TMazeState(1, :south), 2) # south sign
test_obs(TMazeState(5, :south), 3) # corridor
test_obs(TMazeState(11, :south), 4) # junction
test_obs(terminalstate, 5) # terminal

ov = convert_o(Array{Float64}, 1, problem)
@test ov == [1.]
o = convert_o(Int64, ov, problem)
@test o == 1

for s in states(problem)
    v = convert_s(Vector{Float64}, s, problem)
    s2 = convert_s(Union{TerminalState,TMazeState}, v, problem)
    @test s2 == s
end

using POMDPModels
using POMDPTesting
using Test

let
    R = [-1. -100 10; -1 10 -100]

    T = zeros(2,3,2)
    T[:,:,1] = [1. 0.5 0.5; 0 0.5 0.5]
    T[:,:,2] = [0. 0.5 0.5; 1 0.5 0.5]

    O = zeros(2,3,2)
    O[:,:,1] = [0.85 0.5 0.5; 0.15 0.5 0.5] 
    O[:,:,2] = [0.15 0.5 0.5; 0.85 0.5 0.5] 

    pomdp1 = TigerPOMDP()

    pomdp2 = TabularPOMDP(T, R, O, 0.95)

    policy = RandomPolicy(pomdp1, rng=MersenneTwister(2))
    sim = RolloutSimulator(rng=MersenneTwister(3), max_steps=100)
    
    simulate(sim, pomdp1, policy, updater(policy), initialstate_distribution(pomdp1))

    # test generate_o
    o = generate_o(pomdp1, true, MersenneTwister(1))
    @test o == 1
    # test vec
    ov = convert_o(Array{Float64}, true, pomdp1)
    @test ov == [1.]
    o = convert_o(Bool, ov, pomdp1)
    @test o == true

    POMDPTesting.probability_check(pomdp1)
    POMDPTesting.probability_check(pomdp2)
end

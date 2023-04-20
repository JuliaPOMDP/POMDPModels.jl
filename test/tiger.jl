using POMDPModels
using POMDPTools
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

    simulate(sim, pomdp1, policy, updater(policy), initialstate(pomdp1))

    o = last(observations(pomdp1))
    @test o == 1
    # test vec
    ov = convert_o(Array{Float64}, true, pomdp1)
    @test ov == [1.]
    o = convert_o(Bool, ov, pomdp1)
    @test o == true

    @test has_consistent_distributions(pomdp1)
    @test has_consistent_distributions(pomdp2)

    @test reward(pomdp1, TIGER_LEFT, TIGER_OPEN_LEFT) == pomdp1.r_findtiger
    @test reward(pomdp1, TIGER_LEFT, TIGER_OPEN_RIGHT) == pomdp1.r_escapetiger
    @test reward(pomdp1, TIGER_RIGHT, TIGER_OPEN_RIGHT) == pomdp1.r_findtiger
    @test reward(pomdp1, TIGER_RIGHT, TIGER_OPEN_LEFT) == pomdp1.r_escapetiger
    @test reward(pomdp1, TIGER_RIGHT, TIGER_LISTEN) == pomdp1.r_listen

    for s in states(pomdp1)
        @test pdf(transition(pomdp1, s, TIGER_LISTEN), s) == 1.0
        @test pdf(transition(pomdp1, s, TIGER_OPEN_LEFT), s) == 0.5
        @test pdf(transition(pomdp1, s, TIGER_OPEN_RIGHT), s) == 0.5
    end

    for s in states(pomdp1)
        @test pdf(observation(pomdp1, TIGER_LISTEN, s), s) == pomdp1.p_listen_correctly
        @test pdf(observation(pomdp1, TIGER_OPEN_LEFT, s), s) == 0.5
        @test pdf(observation(pomdp1, TIGER_OPEN_RIGHT, s), s) == 0.5
    end


end

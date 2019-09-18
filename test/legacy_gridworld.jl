using POMDPModels
using Test
using POMDPTesting

let
    problem = LegacyGridWorld()

    policy = RandomPolicy(problem)

    sim = HistoryRecorder(rng=MersenneTwister(1), max_steps=1000)

    hist = simulate(sim, problem, policy, GridWorldState(1,1))

    for (s, a) in zip(state_hist(hist), action_hist(hist))
        td = transition(problem, s, a)
        @test sum(td.probs) ≈ 1.0 atol=0.01
        for p in td.probs
            @test p >= 0.0
        end
    end

    sv = convert_s(Array{Float64}, GridWorldState(1, 1, false), problem)
    @test sv == [1.0, 1.0, 0.0]
    sv = convert_s(Array{Float64}, GridWorldState(5, 3, false), problem)
    @test sv == [5.0, 3.0, 0.0]
    s = convert_s(GridWorldState, sv, problem)
    @test s == GridWorldState(5, 3, false)

    av = convert_a(Array{Float64}, :up, problem)
    @test av == [0.0]
    a = convert_a(Symbol, av, problem)
    @test a == :up

    @test GridWorldState(1,1,false) == GridWorldState(1,1,false)
    @test hash(GridWorldState(1,1,false)) == hash(GridWorldState(1,1,false))
    @test GridWorldState(1,2,false) != GridWorldState(1,1,false)
    @test GridWorldState(1,2,true) == GridWorldState(1,1,true)
    @test hash(GridWorldState(1,2,true)) == hash(GridWorldState(1,1,true))

    POMDPTesting.trans_prob_consistency_check(problem)

    # test gridworld deprecation - these can be removed once the deprecation period is over
    @test GridWorld() isa LegacyGridWorld
    @test GridWorld(sx=9) isa LegacyGridWorld
    @test GridWorld(9, 9, tp=0.8) isa LegacyGridWorld
end

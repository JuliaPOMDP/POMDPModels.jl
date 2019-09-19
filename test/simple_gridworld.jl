using POMDPModels
using Test
using POMDPTesting
using NBInclude

let
    problem = SimpleGridWorld()

    policy = RandomPolicy(problem)

    sim = HistoryRecorder(rng=MersenneTwister(1), max_steps=1000)

    hist = simulate(sim, problem, policy, GWPos(1,1))

    for (s, a) in zip(state_hist(hist), action_hist(hist))
        td = transition(problem, s, a)
        if td isa SparseCat
            @test sum(td.probs) ≈ 1.0 atol=0.01
            for p in td.probs
                @test p >= 0.0
            end
        end
    end

    sv = convert_s(Array{Float64}, GWPos(1,1), problem)
    @test sv == [1.0, 1.0]
    sv = convert_s(Array{Float64}, GWPos(5,3), problem)
    @test sv == [5.0, 3.0]
    s = convert_s(GWPos, sv, problem)
    @test s == GWPos(5, 3)

    av = convert_a(Array{Float64}, :up, problem)
    @test av == [1.0]
    a = convert_a(Symbol, av, problem)
    @test a == :up

    POMDPTesting.trans_prob_consistency_check(problem)

    pol = FunctionPolicy(x->:up)
    stp = first(stepthrough(problem, pol, "s,a", max_steps=1))
    POMDPModelTools.render(problem, stp)
    POMDPModelTools.render(problem, NamedTuple())
    POMDPModelTools.render(problem, stp, color=s->reward(problem,s))
    POMDPModelTools.render(problem, stp, color=s->rand())
    POMDPModelTools.render(problem, stp, color=s->"yellow")

    ss = collect(states(problem))
    isd = initialstate_distribution(problem)
    for s in ss
        if !isterminal(problem, s)
            @test s in support(isd)
            @test pdf(isd, s) > 0.0
        end
    end
end

let
    @nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "GridWorld Visualization.ipynb"))
end

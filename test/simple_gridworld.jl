using POMDPModels
using POMDPTools
using Test

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

    @test has_consistent_transition_distributions(problem)

    pol = FunctionPolicy(x->:up)
    stp = first(stepthrough(problem, pol, "s,a", max_steps=1))
    POMDPTools.render(problem, stp)
    POMDPTools.render(problem, NamedTuple())
    POMDPTools.render(problem, stp, color=s->reward(problem,s))
    POMDPTools.render(problem, stp, color=s->rand())
    POMDPTools.render(problem, stp, color=s->"yellow")
    POMDPTools.render(problem, stp, color=s->reward(problem,s), colormin=-1.0, colormax=1.0)

    ss = collect(states(problem))
    isd = initialstate(problem)
    for s in ss
        if !isterminal(problem, s)
            @test s in support(isd)
            @test pdf(isd, s) > 0.0
        end
    end
end

let
    @warn("NBInclude tests skipped")
    # @nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "GridWorld Visualization.ipynb"))
end

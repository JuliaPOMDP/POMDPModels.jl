using POMDPModels
using POMDPTools
using Test

function gw_slow_transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + POMDPModels.dir[act]
        destinations[i+1] = dest

        if !POMDPModels.inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end

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

    @testset "fast transition consistency" begin
        for s ∈ states(problem)
            for a ∈ actions(problem)
                t1 = transition(problem, s, a)
                t2 = gw_slow_transition(problem, s, a)
                for sp ∈ states(problem)
                    @test pdf(t1, sp) ≈ pdf(t2, sp)
                end
            end
        end
    end
end

let
    @warn("NBInclude tests skipped")
    # @nbinclude(joinpath(dirname(@__FILE__), "..", "notebooks", "GridWorld Visualization.ipynb"))
end

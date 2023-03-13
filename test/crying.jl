let
    problem = BabyPOMDP()

    # starve policy
    # when the baby is never fed, the reward for starting in the hungry state should be -100
    sim = RolloutSimulator(eps=0.0001)
    ib = nothing
    policy = Starve()
    r = simulate(sim, problem, policy, updater(policy), ib, true)
    @test r ≈ -100.0 atol=0.01

    # test vec
    ov = convert_s(Array{Float64}, true, problem)
    @test ov == [1.]
    o = convert_s(Bool, ov, problem)
    @test o == true

    @test has_consistent_distributions(problem)

    bu = DiscreteUpdater(problem)
    bp =  update(bu,
                 initialize_belief(bu, BoolDistribution(0.0)),
                 false,
                 true)

    @test pdf(bp, true) ≈ 0.47058823529411764 atol=0.0001
    r = simulate(sim, problem, policy, DiscreteUpdater(problem), BoolDistribution(1.0))
    @test r ≈ -100.0 atol=0.01
end

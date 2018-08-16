using POMDPModels
using POMDPs

let
    problem = BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.9)

    n = 100000
    results = Vector{Float64}(n)

    i = 1
    rng = MersenneTwister(i)
    init_state = false
    od = observation(problem, init_state, false, init_state)
    obs = rand(rng, od)
    sim = RolloutSimulator(rng=rng, initialstate=init_state, eps=0.0001)
    policy = FeedWhenCrying()
    results[i] = simulate(sim, problem, policy, updater(policy), PreviousObservation(obs))

    rngs = [MersenneTwister(i) for i in 1:n]

    @time for i in 1:n
        rng = rngs[i]
        init_state = false
        od = observation(problem, init_state, false, init_state)
        obs = rand(rng, od)
        sim = RolloutSimulator(rng=rng, initialstate=init_state, eps=0.0001)
        policy = FeedWhenCrying()
        results[i] = simulate(sim, problem, policy, updater(policy), PreviousObservation(obs))
    end
end

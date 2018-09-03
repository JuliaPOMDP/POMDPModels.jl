# test simple policies on the crying baby problem

using Test

using POMDPModels
using POMDPs

problem = BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.9)

# starve policy
# when the baby is never fed, the reward for starting in the hungry state should be -100
sim = RolloutSimulator()
sim.eps = 0.0001
sim.initialstate = true
ib = EmptyBelief()
policy = Starve()
r = simulate(sim, problem, policy, updater(policy), ib)
@test r ≈ -100.0 atol=0.01

# when the baby is never fed the average reward for starting in the full state should be -47.37
# should take ~5 seconds
# by the Hoeffding inequality there is less than a 1.3% chance that the average will be off by 0.5
n = 100000
r_sum = @parallel (+) for i in 1:n
    sim = RolloutSimulator(MersenneTwister(i),
                           false,
                           0.001,
                           nothing)
    policy = Starve()
    simulate(sim, problem, policy, updater(policy), EmptyBelief())
end
@test r_sum/n ≈ -47.37 atol=0.5

# always feed policy
# when the baby is always fed the reward for starting in the full state should be -50
sim = RolloutSimulator(MersenneTwister(), false, 0.0001, nothing)
policy = AlwaysFeed()
r = simulate(sim, problem, policy, updater(policy), EmptyBelief())
@test r ≈ -50.0 atol=0.01

# when the baby is always fed the reward for starting in the hungry state should be -60
sim = RolloutSimulator(MersenneTwister(), true, 0.0001, nothing)
policy = AlwaysFeed()
r = simulate(sim, problem, policy, updater(policy), EmptyBelief())
@test r ≈ -60.0 atol=0.01

# println("finished easy tests")

# good policy - feed when the last observation was crying - this is *almost* optimal
# from full state, reward should be -17.14
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = false
    od = observation(problem, init_state, false, init_state)
    obs = rand(rng, od)
    sim = RolloutSimulator(rng=rng, initialstate=init_state, eps=0.0001)
    policy = FeedWhenCrying()
    simulate(sim, problem, policy, updater(policy), PreviousObservation(obs))
    # println(i)
end
@test r_sum/n ≈ -17.14 atol=0.1

# from hungry state, reward should be -32.11
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = true
    od = observation(problem, init_state, false, init_state)
    obs = rand(rng, od)
    sim = RolloutSimulator(rng=rng, initialstate=init_state, eps=0.0001)
    policy = FeedWhenCrying()
    simulate(sim, problem, policy, updater(policy), PreviousObservation(obs))
end
@test r_sum/n ≈ -32.11 atol=0.1

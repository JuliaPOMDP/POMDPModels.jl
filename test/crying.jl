# test simple policies on the crying baby problem

using Base.Test

using POMDPModels
using POMDPToolbox
using POMDPs

problem = BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.9)

# starve policy
# when the baby is never fed, the reward for starting in the hungry state should be -100
sim = RolloutSimulator()
sim.eps = 0.0001
sim.initial_state = BabyState(true)
sim.initial_belief = EmptyBelief()
r = simulate(sim, problem, Starve())
@test_approx_eq_eps r -100.0 0.01

# when the baby is never fed the average reward for starting in the full state should be -47.37
# should take ~5 seconds
# by the Hoeffding inequality there is less than a 1.3% chance that the average will be off by 0.5
n = 100000
r_sum = @parallel (+) for i in 1:n
    sim = RolloutSimulator(MersenneTwister(i),
                           EmptyBelief(),
                           BabyState(false),
                           0.001,
                           nothing)
    simulate(sim, problem, Starve())
end
@test_approx_eq_eps r_sum/n -47.37 0.5

# always feed policy
# when the baby is always fed the reward for starting in the full state should be -50
sim = RolloutSimulator(MersenneTwister(), EmptyBelief(), BabyState(false), 0.0001, nothing)
r = simulate(sim, problem, AlwaysFeed())
@test_approx_eq_eps r -50.0 0.01

# when the baby is always fed the reward for starting in the hungry state should be -60
sim = RolloutSimulator(MersenneTwister(), EmptyBelief(), BabyState(true), 0.0001, nothing)
r = simulate(sim, problem, AlwaysFeed())
@test_approx_eq_eps r -60.0 0.01

# println("finished easy tests")

# good policy - feed when the last observation was crying - this is *almost* optimal
# from full state, reward should be -17.14
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = BabyState(false)
    obs = create_observation(problem)
    od = observation(problem, init_state, BabyAction(true))
    rand!(rng, obs, od)
    sim = RolloutSimulator(rng, PreviousObservation(obs), init_state, 0.0001, nothing)
    simulate(sim, problem, FeedWhenCrying())
    # println(i)
end
@test_approx_eq_eps r_sum/n -17.14 0.1

# from hungry state, reward should be -32.11
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = BabyState(true)
    obs = create_observation(problem)
    od = observation(problem, init_state, BabyAction(true))
    rand!(rng, obs, od)
    sim = RolloutSimulator(rng, PreviousObservation(obs), init_state, 0.0001, nothing)
    simulate(sim, problem, FeedWhenCrying())
end
@test_approx_eq_eps r_sum/n -32.11 0.1



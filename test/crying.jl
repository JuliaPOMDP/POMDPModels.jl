# test simple policies on the crying baby problem

using Base.Test

using POMDPModels
using POMDPToolbox
using POMDPs

problem = BabyPOMDP(-1, -10, 0.1, 0.8, 0.1, 0.9)

# starve policy
# when the baby is never fed, the reward for starting in the hungry state should be -100
r = simulate(problem, Starve(), EmptyBelief(), eps=0.0001, initial_state=BabyState(true))
@test_approx_eq_eps r -100.0 0.01

# when the baby is never fed the average reward for starting in the full state should be -47.37
# should take ~5 seconds
# by the Hoeffding inequality there is less than a 1.3% chance that the average will be off by 0.5
n = 100000
r_sum = @parallel (+) for i in 1:n
    simulate(problem,
             Starve(),
             EmptyBelief(),
             eps=0.001,
             rng=MersenneTwister(i),
             initial_state=BabyState(false))
end
@test_approx_eq_eps r_sum/n -47.37 0.5

# always feed policy
# when the baby is always fed the reward for starting in the full state should be -10
r = simulate(problem, AlwaysFeed(), EmptyBelief(), eps=0.0001, initial_state=BabyState(false))
@test_approx_eq_eps r -10.0 0.01

# when the baby is always fed the reward for starting in the hungry state should be -20
r = simulate(problem, AlwaysFeed(), EmptyBelief(), eps=0.0001, initial_state=BabyState(true))
@test_approx_eq_eps r -20.0 0.01

# good policy - feed when the last observation was crying - this is *almost* optimal
# from full state, reward should be -10.62
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = BabyState(false)
    obs = create_observation(problem)
    od = create_observation_distribution(problem)
    observation!(od, problem, init_state, BabyAction(true))
    rand!(rng, obs, od)
    simulate(problem,
             FeedWhenCrying(),
             PreviousObservation(obs),
             eps=0.0001,
             rng=rng,
             initial_state=init_state)
end
@test_approx_eq_eps r_sum/n -10.62 0.1

# from hungry state, reward should be -22.5
n = 100000
r_sum = @parallel (+) for i in 1:n
    rng = MersenneTwister(i)
    init_state = BabyState(true)
    obs = create_observation(problem)
    od = create_observation_distribution(problem)
    observation!(od, problem, init_state, BabyAction(true))
    rand!(rng, obs, od)
    simulate(problem,
             FeedWhenCrying(),
             PreviousObservation(obs),
             eps=0.0001,
             rng=rng,
             initial_state=init_state)
end
@test_approx_eq_eps r_sum/n -22.50 0.1



using POMDPModels
using POMDPToolbox
using Base.Test
using NBInclude

problem = GridWorld()

policy = RandomPolicy(problem)

sim = HistoryRecorder(rng=MersenneTwister(1), max_steps=1000)

simulate(sim, problem, policy, GridWorldState(1,1))

for i in 1:length(sim.action_hist)
    td = transition(problem, sim.state_hist[i], sim.action_hist[i])
    @test_approx_eq_eps sum(td.probs) 1.0 0.01
    for p in td.probs
        @test p >= 0.0
    end
end


sv = convert(Array{Float64}, GridWorldState(1, 1, false), problem)
@test sv == [1.0, 1.0, 0.0]
sv = convert(Array{Float64}, GridWorldState(5, 3, false), problem)
@test sv == [5.0, 3.0, 0.0]
s = convert(GridWorldState, sv, problem)
@test s == GridWorldState(5, 3, false)

av = convert(Array{Float64}, :up, problem)
@test av == [0.0]
a = convert(Symbol, av, problem)
@test a == :up

@test GridWorldState(1,1,false) == GridWorldState(1,1,false)
@test hash(GridWorldState(1,1,false)) == hash(GridWorldState(1,1,false))
@test GridWorldState(1,2,false) != GridWorldState(1,1,false)
@test GridWorldState(1,2,true) == GridWorldState(1,1,true)
@test hash(GridWorldState(1,2,true)) == hash(GridWorldState(1,1,true))

trans_prob_consistency_check(problem)

nbinclude(joinpath(Pkg.dir("POMDPModels"), "notebooks", "GridWorld Visualization.ipynb"))

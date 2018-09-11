using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using BeliefUpdaters
using BenchmarkTools

m = LightDark1D()
struct P <: Policy end
POMDPs.action(::P,::Any) = rand(-1:2:1)
POMDPs.updater(::P) = NothingUpdater()
p = P()
sim = RolloutSimulator(max_steps=100)

@btime simulate($sim, $m, $p)

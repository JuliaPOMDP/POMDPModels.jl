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
sim = RolloutSimulator(max_steps=1000)

@btime simulate($sim, $m, $p)

# Output from old version with LightDark1DState
#= 
julia> include("benchmarks/lightdark.jl")
  23.077 μs (1 allocation: 8 bytes)
0.0
=#

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
# Output from new_lightdark version with terminal states
#=
julia> include("benchmarks/lightdark.jl")
  225.417 μs (6001 allocations: 109.38 KiB)
0.0
=#
# Output from nan_lightdark version with NaN as terminal state
#=
julia> include("benchmarks/lightdark.jl")
  23.438 μs (1 allocation: 8 bytes)
0.0
=#

# using Profile
# using ProfileView
# sim = RolloutSimulator(max_steps=1000)
# simulate(sim, m, p)
# Profile.clear()
# @profile for i in 1:1000
#     simulate(sim, m, p)
# end
# ProfileView.view()



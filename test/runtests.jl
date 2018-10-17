using POMDPModels
using POMDPSimulators
using POMDPModelTools
using BeliefUpdaters
using POMDPPolicies
using Random
using Test
using Compose

@testset "crying" begin
    include("crying.jl")
end
@testset "gridworld" begin
    include("simple_gridworld.jl")
end
@testset "gridworld_vis" begin
    include("test_gw_vis.jl")
end
@testset "legacy_gridworld" begin
    include("legacy_gridworld.jl")
end
@testset "tiger" begin
    include("tiger.jl")
end
@testset "random" begin
    include("random.jl")
end
@testset "tmaze" begin
    include("tmaze.jl")
end
@testset "inverted" begin
    include("inverted.jl")
end
@testset "car" begin
    include("car.jl")
end
@testset "lightdark" begin
    include("lightdark.jl")
end

using POMDPModels
using POMDPSimulators
using POMDPModelTools
using BeliefUpdaters
using POMDPPolicies
using Random
using Test

@testset "crying" begin
    include("crying.jl")
end
@testset "gridworld" begin
    include("gridworld.jl")
end
@testset "tiger" begin
    include("tiger.jl")
end
@testset "random" begin
    include("random.jl")
end
# # include("tmaze.jl")
@testset "inverted" begin
    include("inverted.jl")
end
@testset "car" begin
    include("car.jl")
end
@testset "lightdark" begin
    include("lightdark.jl")
end

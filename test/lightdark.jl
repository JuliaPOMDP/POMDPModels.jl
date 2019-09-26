using Test
using POMDPModels

let
    rng = MersenneTwister(41)
    p = LightDark1D()
    @test discount(p) == 0.9
    s0 = LightDark1DState(0,0)
    s0, _, r = gen(DDNOut(:sp, :o, :r), p, s0, +1, rng)
    @test s0.y == 1.0
    @test r == 0
    s1, _, r = gen(DDNOut(:sp, :o, :r), p, s0, 0, rng)
    @test s1.status != 0
    @test r == -10.0
    s2 = LightDark1DState(0, 5)
    obs = gen(DDNNode(:o), p, nothing, nothing, s2, rng)
    @test abs(obs-6.0) <= 1.1


    sv = convert_s(Array{Float64}, s2, p)
    @test sv == [0.0, 5.0]
    s = convert_s(LightDark1DState, sv, p)
    @test s == s2

    ov = convert_o(Array{Float64}, obs, p)
    @test ov == [obs]
    o = convert_o(Float64, ov, p)
    @test o == obs
end

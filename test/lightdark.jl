using Test
using POMDPModels

let
    rng = MersenneTwister(41)
    p = LightDark1D()
    @test discount(p) == 0.9
    s0 = 0.0
    s0, _, r = generate_sor(p, s0, +1, rng)
    @test s0 == 1.0
    @test r == 0
    s1, _, r = generate_sor(p, s0, 0, rng)
    @test isterminal(p, s1)
    @test r == -10.0
    s2 = 5.0
    obs = generate_o(p, nothing, nothing, s2, rng)
    @test abs(obs-6.0) <= 1.1

    # this might be a problem
    sv = convert_s(Array{Float64}, s2, p)
    @test sv == [5.0]
    s = convert_s(Float64, sv, p)
    @test s == s2

    ov = convert_o(Array{Float64}, obs, p)
    @test ov == [obs]
    o = convert_o(Float64, ov, p)
    @test o == obs
end

using Base.Test
using POMDPModels

rng = MersenneTwister(41)
p = LightDark1D()
@test discount(p) == 0.9
s0 = LightDark1DState(0,0)
s0, _, r = generate_sor(p, s0, +1, rng)
@test s0.y == 1.0
@test r == 0
s1, _, r = generate_sor(p, s0, 0, rng)
@test s1.status != 0
@test r == -10.0
s2 = LightDark1DState(0, 5)
obs = generate_o(p, nothing, nothing, s2, rng)
@test abs(obs-6.0) <= 1.1

ov = convert(p, obs)
@test ov == [obs]
o = convert(p, ov)
@test o == obs

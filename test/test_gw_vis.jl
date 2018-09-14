using POMDPPolicies
using POMDPModels
using Compose
using POMDPSimulators

m = SimpleGridWorld()
p = FunctionPolicy(s->:up)
is = GWPos(3,4)
stage = first(stepthrough(m, p, is, max_steps=1))

c = POMDPModels.render(m, stage)
draw(SVG(tempname()*".svg", 10cm, 10cm), c)

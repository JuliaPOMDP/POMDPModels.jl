m = SimpleGridWorld()
p = FunctionPolicy(s->:up)
is = GWPos(3,4)
stage = first(stepthrough(m, p, is, max_steps=1))

c = POMDPModels.render(m, stage)
draw(SVG(tempname()*".svg", 10cm, 10cm), c)

c = POMDPModels.render(m, stage, color=s->reward(m,s))
draw(SVG(tempname()*".svg", 10cm, 10cm), c)

c = POMDPModels.render(m, stage, color=s->rand())
draw(SVG(tempname()*".svg", 10cm, 10cm), c)

c = POMDPModels.render(m, stage, color=s->"yellow")
draw(SVG(tempname()*".svg", 10cm, 10cm), c)

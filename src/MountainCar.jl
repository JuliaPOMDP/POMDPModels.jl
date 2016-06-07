# Mountain Car problem for continuous reinforcement learning
# As described in XXX

type MountainCar <: MDP{Tuple{Float64,Float64},Float64}
  discount::Float64
  cost::Float64
end
MountainCar(;discount::Float64=0.99,cost::Float64=-1.) = MountainCar(discount,cost)

POMDPs.create_state(::MountainCar) = (-0.5,0.,)
POMDPs.create_action(::MountainCar) = 0.

type MountainCarActions <: AbstractSpace
  actions::Vector{Float64}
end
MountainCarActions() = MountainCarActions(Float64[-1.,0.,1.])
POMDPs.actions(::MountainCar) = MountainCarActions(Float64[-1.,0.,1.])

POMDPs.reward(mc::MountainCar,
              s::Tuple{Float64,Float64},
              a::Float64,
              sp::Tuple{Float64,Float64}) = isterminal(mc,s) ? 0. : mc.cost

function GenerativeModels.initial_state(mc::MountainCar,
                                        ::AbstractRNG,
                                        sp::Tuple{Float64,Float64}=create_state(mc))
  sp = (-0.5,0.,)
  return sp
end

POMDPs.isterminal(::MountainCar,s::Tuple{Float64,Float64}) = s[1] >= 0.5
POMDPs.discount(mc::MountainCar) = mc.discount

function GenerativeModels.generate_s( mc::MountainCar,
                                      s::Tuple{Float64,Float64},
                                      a::Float64,
                                      ::AbstractRNG,
                                      sp::Tuple{Float64,Float64} = create_state(mc))
  x,v = s
  v_ = v + a*0.001+cos(3*x)*-0.0025
  v_ = max(min(0.07,v_),-0.07)
  x_ = x+v_
  #inelastic boundary
  if x_ < -1.2
      x_ = -1.2
      v_ = 0.
  end
  sp = (x_,v_,)
  return sp
end

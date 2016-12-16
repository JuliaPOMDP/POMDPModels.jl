# Mountain Car problem for continuous reinforcement learning
# As described in XXX

type MountainCar <: MDP{Tuple{Float64,Float64},Float64}
  discount::Float64
  cost::Float64 # reward at each state not at the goal (should be a negative number)
  jackpot::Float64 # reward at the top
  vec_state::Vector{Float64}
end
MountainCar(;discount::Float64=0.99,cost::Float64=-1., jackpot::Float64=0.0) = MountainCar(discount,cost,jackpot,zeros(2))

type MountainCarActions <: AbstractSpace
  actions::Vector{Float64}
end
MountainCarActions() = MountainCarActions(Float64[-1.,0.,1.])
actions(::MountainCar) = MountainCarActions(Float64[-1.,0.,1.])
actions(mc::MountainCar,::Tuple{Float64,Float64}) = actions(mc)
n_actions(mc::MountainCar) = 3
rand(rng::AbstractRNG,as::MountainCarActions) = as.actions[rand(rng,1:length(as.actions))]

reward(mc::MountainCar,
              s::Tuple{Float64,Float64},
              a::Float64,
              sp::Tuple{Float64,Float64}) = isterminal(mc,sp) ? mc.jackpot : mc.cost

function initial_state(mc::MountainCar, ::AbstractRNG)
  sp = (-0.5,0.,)
  return sp
end

isterminal(::MountainCar,s::Tuple{Float64,Float64}) = s[1] >= 0.5
discount(mc::MountainCar) = mc.discount

function generate_s( mc::MountainCar,
                     s::Tuple{Float64,Float64},
                     a::Float64,
                     ::AbstractRNG)
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


function vec(mc::MountainCar, s::Tuple{Float64,Float64})
    copy!(mc.vec_state, s)
    return mc.vec_state
end

# Example policy -- works pretty well
type Energize <: Policy{Tuple{Float64,Float64}} end
action(::Energize,s::Tuple{Float64,Float64}) = sign(s[2])

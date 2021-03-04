# Mini Hallway problem defined in http://cs.brown.edu/research/ai/pomdp/examples/mini-hall2.POMDm.
# Original idea published in Littman, Cassandra and Kaelbling's ML-95 paper.

# Basic parameters are:
# discount: 0.950000
# values: reward
# states: 13
# actions: 3
# observations: 9
# The rest is available at link


using POMDPs
using SparseArrays
using POMDPModelTools
using POMDPLinter

struct Observation
    no::Int
    prob::Float64
end

struct MiniHallway <: POMDP{Int, Int, Observation}
    T::Array{Float64}
    R::Array{Int64}
    O::Array{Float64, 2}
    O_prob::Array{Observation}
    discount::Float64
end

function MiniHallway()
    T = Array{Float64, 3}(undef, 13, 3, 13)
    T[:, 1, :] = sparse(collect(1:12), [1, 2, 7, 4, 1, 10, 1, 8, 9, 10, 13, 8], ones(12), 13, 13)
    T[:, 2, :] = sparse(collect(1:12), [2, 3, 4, 1, 6, 7, 8, 5, 10, 11, 12, 9], ones(12), 13, 13)
    T[:, 3, :] = sparse(collect(1:12), [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11], ones(12), 13, 13)
    T[12, :, :] =  hcat([[0.083337, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.] for i in 1:3]...)'

    discount = 0.95
    R = zeros(13)
    R[13] = 1

    # 13 x 9 matrix, access through O[:, O_no]
    O = Array(sparse(collect(1:13), [1, 2, 3, 4, 5, 6, 7, 8, 7, 8, 5, 6, 9], [1., 1., 1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.]))
    O_prob = Array{Observation}(undef, 13)
    O_prob[1] = Observation(1, 1.); O_prob[2] = Observation(2, 1.); O_prob[3] = Observation(3, 1.);
    O_prob[4] = Observation(4, 1.); O_prob[5] = Observation(5, 1.); O_prob[6] = Observation(6, 1.);
    O_prob[7] = Observation(7, 1.); O_prob[8] = Observation(8, 1.); O_prob[9] = Observation(7, 1.);
    O_prob[10] = Observation(8, 1.); O_prob[11] = Observation(5, 1.); O_prob[12] = Observation(6, 1.);
    O_prob[13] = Observation(9, 1.);

    return MiniHallway(T, R, O, O_prob, discount)
end

##################
# mdps interface #
##################

POMDPs.initialstate

function POMDPs.states(m::MiniHallway)::Array{Int, 1}
    return collect(1:13)
end

POMDPs.stateindex(m::MiniHallway, ss::Int) = ss
POMDPs.isterminal(m::MiniHallway, ss::Int) = ss == 13

function POMDPs.transition(m::MiniHallway, ss::Int, a::Int)
    sps = findall(x -> x > 0., m.T[ss, a, :])
    probs = m.T[ss, a, sps]
    return SparseCat(sps, probs)
end

POMDPs.actions(m::MiniHallway)::Array{Int, 1} = collect(1:3)
POMDPs.actionindex(m::MiniHallway, a::Int)::Int = a

POMDPs.reward(m::MiniHallway, ss::Int, a::Int, sp::Int)::Float64 = m.R[sp]
POMDPs.discount(m::MiniHallway)::Float64 = m.discount

####################
# pomdps interface #
####################

POMDPs.initialstate(m::MiniHallway) = m.T[12, 1, :]
POMDPs.observations(m::MiniHallway) = collect([Observation(i, -1.) for i in 1:9])
POMDPs.obsindex(m::MiniHallway, o::Observation)::Int = o.no
POMDPs.obsindex(m::MiniHallway, o::Int)::Int = o

function POMDPs.observation(m::MiniHallway, sp::Int)::SparseCat
    obs_no = m.O_prob[sp].no
    return SparseCat(states(m), m.O[:, obs_no])
end

function POMDPs.pdf(o::Array{Float64}, ss::Int) 
    return o[ss]
end




# Mini Hallway problem defined in http://cs.brown.edu/research/ai/pomdp/examples/mini-hall2.POMDP.
# Original idea published in Littman, Cassandra and Kaelbling's ML-95 paper.

# Basic parameters are:
# discount: 0.950000
# values: reward
# states: 13
# actions: 3
# observations: 9
# The rest is available at link


struct Observation
    no::Int
end

struct MiniHallway <: POMDP{Int, Int, Observation}
    T::Array{SparseCat, 2}
    R::Array{Float64}
    O::Array{SparseCat{Array{Observation, 1}, Array{Float64, 1}}}
    discount::Float64
end

function MiniHallway()
    T = Array{SparseCat, 2}(undef, 13, 3)

    T[1,1] = SparseCat((1), (1.)); T[1,2] = SparseCat((2), (1.)); T[1,3] = SparseCat((4), (1.))
    T[2,1] = SparseCat((1), (1.)); T[2,2] = SparseCat((3), (1.)); T[2,3] = SparseCat((1), (1.))
    T[3,1] = SparseCat((7), (1.)); T[3,2] = SparseCat((4), (1.)); T[3,3] = SparseCat((2), (1.))
    T[4,1] = SparseCat((4), (1.)); T[4,2] = SparseCat((1), (1.)); T[4,3] = SparseCat((3), (1.))
    T[5,1] = SparseCat((1), (1.)); T[5,2] = SparseCat((6), (1.)); T[5,3] = SparseCat((8), (1.))
    T[6,1] = SparseCat((10), (1.)); T[6,2] = SparseCat((7), (1.)); T[6,3] = SparseCat((5), (1.))
    T[7,1] = SparseCat((1), (1.)); T[7,2] = SparseCat((8), (1.)); T[7,3] = SparseCat((6), (1.))
    T[8,1] = SparseCat((8), (1.)); T[8,2] = SparseCat((5), (1.)); T[8,3] = SparseCat((7), (1.))
    T[9,1] = SparseCat((9), (1.)); T[9,2] = SparseCat((10), (1.)); T[9,3] = SparseCat((12), (1.))
    T[10,1] = SparseCat((10), (1.)); T[10,2] = SparseCat((11), (1.)); T[10,3] = SparseCat((9), (1.))
    T[11,1] = SparseCat((13), (1.)); T[11,2] = SparseCat((12), (1.)); T[11,3] = SparseCat((10), (1.))
    T[12,1] = SparseCat((8), (1.)); T[12,2] = SparseCat((9), (1.)); T[12,3] = SparseCat((11), (1.))
    T[13,1] = SparseCat((1:13), (0.083337, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.))
    T[13,2] = SparseCat((1:13), (0.083337, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.))
    T[13,3] = SparseCat((1:13), (0.083337, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.083333, 0.))
    discount = 0.95
    R = zeros(13)
    R[13] = 1.

    O = Array{SparseCat}(undef, 13)
    O[1] = SparseCat([Observation(1)], [1.]); O[2] = SparseCat([Observation(2)], [1.]); O[3] = SparseCat([Observation(3)], [1.]);
    O[4] = SparseCat([Observation(4)], [1.]); O[5] = SparseCat([Observation(5)], [1.]); O[6] = SparseCat([Observation(6)], [1.]);
    O[7] = SparseCat([Observation(7)], [1.]); O[8] = SparseCat([Observation(8)], [1.]); O[9] = SparseCat([Observation(7)], [1.]);
    O[10] = SparseCat([Observation(8)], [1.]); O[11] = SparseCat([Observation(5)], [1.]); O[12] = SparseCat([Observation(6)], [1.]);
    O[13] = SparseCat([Observation(9)], [1.]);

    return MiniHallway(T, R, O, discount)
end

##################
# mdps interface #
##################
function POMDPs.states(m::MiniHallway)::Array{Int, 1}
    return 1:13
end

POMDPs.stateindex(m::MiniHallway, ss::Int) = ss
POMDPs.isterminal(m::MiniHallway, ss::Int) = ss == 13

function POMDPs.transition(m::MiniHallway, ss::Int, a::Int)
    return m.T[ss, a]
end
POMDPs.transition(m::MiniHallway, ss::Int, a::Int, sp::Int) = transition(m, ss, a)

POMDPs.actions(m::MiniHallway)::Array{Int, 1} = 1:3
POMDPs.actionindex(m::MiniHallway, a::Int)::Int = a

POMDPs.reward(m::MiniHallway, ss::Int, a::Int, sp::Int)::Float64 = m.R[sp]
POMDPs.discount(m::MiniHallway)::Float64 = m.discount

####################
# pomdps interface #
####################
POMDPs.initialstate(m::MiniHallway) = SparseCat(1:13, collect(m.T[13, 1].probs))
POMDPs.observations(m::MiniHallway) = (Observation(i) for i in 1:9)
POMDPs.obsindex(m::MiniHallway, o::Observation)::Int = o.no
POMDPs.obsindex(m::MiniHallway, o::Int)::Int = o

function POMDPs.observation(m::MiniHallway, a::Int, sp::Int)::SparseCat
    return m.O[sp]
end
POMDPs.observation(m::MiniHallway, s::Int, a::Int, sp::Int)::SparseCat = observation(m, a, sp)

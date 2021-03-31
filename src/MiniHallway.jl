# Mini Hallway problem defined in http://cs.brown.edu/research/ai/pomdp/examples/mini-hall2.POMDP.
# Original idea published in Littman, Cassandra and Kaelbling's ML-95 paper.

# Basic parameters are:
# discount: 0.950000
# values: reward
# states: 13
# actions: 3
# observations: 9
# The rest is available at link


struct MiniHallway <: POMDP{Int, Int, Int}
    T::Array{Union{Deterministic, DiscreteUniform}, 1}
end

function MiniHallway()
    T = Array{Union{Deterministic, DiscreteUniform}, 1}(undef, 13)

    # Transitions for action 1 (and all actions in state 13) as I did not find a function for it
    T[1] = Deterministic(1); T[2] = Deterministic(2); T[3] = Deterministic(7);
    T[4] = Deterministic(4); T[5] = Deterministic(1); T[6] = Deterministic(10);
    T[8] = Deterministic(8); T[7] = Deterministic(1); T[9] = Deterministic(9);
    T[10] = Deterministic(10); T[11] = Deterministic(13); T[12] = Deterministic(8);
    T[13] = DiscreteUniform(1, 12)

    return MiniHallway(T)
end

##################
# mdps interface #
##################
POMDPs.states(m::MiniHallway) = 1:13
POMDPs.stateindex(m::MiniHallway, ss::Int)::Int = ss
POMDPs.isterminal(m::MiniHallway, ss::Int)::Bool = ss == 13

function POMDPs.transition(m::MiniHallway, ss::Int, a::Int)
    if a == 1 || ss == 13
        return m.T[ss]
    else
        if a == 2
            return ss % 4 == 0 ? Deterministic(ss - 3) : Deterministic(ss + 1)
        elseif a == 3
            return (ss - 1) % 4 == 0 ? Deterministic(ss + 3) : Deterministic(ss - 1)
        end
    end
end

POMDPs.actions(m::MiniHallway) = 1:3
POMDPs.actionindex(m::MiniHallway, a::Int)::Int = a

POMDPs.reward(m::MiniHallway, ss::Int, a::Int, sp::Int) = float(sp==13)
POMDPs.discount(m::MiniHallway)::Float64 = 0.95

####################
# pomdps interface #
####################
POMDPs.initialstate(m::MiniHallway) = m.T[13]
POMDPs.observations(m::MiniHallway) = (i for i in 1:9)

function POMDPs.observation(m::MiniHallway, a::Int, sp::Int)::Deterministic
    if sp <= 8
        return Deterministic(sp)
    elseif sp <= 10
        return Deterministic(sp - 2)
    elseif sp <= 12
        return Deterministic(sp - 6)
    else
        return Deterministic(9)
    end

    return m.O[sp]
end
POMDPs.observation(m::MiniHallway, s::Int, a::Int, sp::Int)::Deterministic = observation(m, a, sp)
POMDPs.obsindex(m::MiniHallway, o::Int)::Int = o

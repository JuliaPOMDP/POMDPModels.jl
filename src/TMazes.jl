@with_kw mutable struct TMazeState
    x::Int64 = 1 # position in corridor
    g::Symbol = :north# goal north or south
    term::Bool = false
end

==(s1::TMazeState, s2::TMazeState) = s1.x == s2.x && s1.g == s2.g
hash(s::TMazeState, h::UInt64 = zero(UInt64)) = hash(s.x, hash(s.g, h))
function Base.copy!(s1::TMazeState, s2::TMazeState)
    s1.x = s2.x
    s1.g = s2.g
    s1.term = s2.term
    return s1
end

@with_kw mutable struct TMaze <: POMDP{TMazeState, Int64, Int64}
    n::Int64 = 10 # corridor length
    discount::Float64 = 0.99 # discount factor
end


# state space is length of corr + 3 cells at the end
#                   |G|
# |S| | | | | | | | | |
#                   | |
# depending on where the goal is
function states(maze::TMaze)
    space = TMazeState[]
    for x in 1:(maze.n + 1), g in [:north, :south]
        push!(space, TMazeState(x, g, false))
    end
    push!(space, TMazeState(1,:none,true)) # terminal
    return space
end
# 4 actions: go North, East, South, West (1, 2, 3, 4)
actions(maze::TMaze) = 1:4
# 5 observations: 2 for goal (left or right) + 2 for in corridor or at intersection + 1 term
observations(maze::TMaze) = 1:5

# transition distribution (actions are deterministic)
mutable struct TMazeStateDistribution
    current_state::TMazeState # deterministic
    reset::Bool
    reset_states::Vector{TMazeState}
    reset_probs::Vector{Float64}
end
function create_transition_distribution(::TMaze)
    rs = [TMazeState(1,:north,false), TMazeState(1,:south,false)]
    rp = [0.5, 0.5]
    TMazeStateDistribution(TMazeState(), false, rs, rp)
end
support(d::TMazeStateDistribution) = d.reset ? (return [(d.current_state, 1.0)]) : (return zip(d.reset_states, d.reset_probs))

function pdf(d::TMazeStateDistribution, s::TMazeState)
    if d.reset
        in(s, d.reset_states) ? (return 0.5) : (return 0.0)
    else
        s == d.current_state ? (return 1.0) : (return 0.0)
    end
end
function rand(rng::AbstractRNG, d::TMazeStateDistribution)
    s = TMazeState()
    if d.reset
        rand(rng) < 0.5 ? (copy!(s, d.reset_states[1])) : (copy!(s, d.reset_states[2]))
        return s
    else
        copy!(s, d.current_state)
        return s
    end
end
#rand(rng::AbstractRNG, d::TMazeStateDistribution)

mutable struct TMazeInit
    states::Vector{TMazeState}
    probs::Vector{Float64}
end
support(d::TMazeInit) = zip(d.states, d.probs)
function initialstate_distribution(maze::TMaze)
    s = states(maze)
    ns = length(s)
    p = zeros(ns) .+ 1.0 / (ns-1)
    p[end] = 0.0
    #s1 = TMazeState(1, :north, false)
    #s2 = TMazeState(1, :south, false)
    #d = TMazeInit([s1, s2])
    return TMazeInit(s, p)
end
function rand(rng::AbstractRNG, d::TMazeInit)
    s = TMazeState()
    #idx = nothing
    #rand(rng) < 0.5 ? (idx = 1) : (idx = 2)
    #copy!(s, d.states[idx])
    cat = Weights(d.probs)
    idx = sample(rng, cat)
    copy!(s, d.states[idx])
    return s
end
function pdf(d::TMazeInit, s::TMazeState)
    for i = 1:length(d.states)
        if d.states[i] == s
            return d.probs[i]
        end
    end
    return 0.0
    #in(s, d.states) ? (return 0.5) : (return 0.0)
end

# observation distribution (deterministic)
mutable struct TMazeObservationDistribution
    current_observation::Int64
end
create_observation_distribution(::TMaze) = TMazeObservationDistribution(1)
iterator(d::TMazeObservationDistribution) = [d.current_observation]

pdf(d::TMazeObservationDistribution, o::Int64) = o == d.current_observation ? (return 1.0) : (return 0.0)
rand(rng::AbstractRNG, d::TMazeObservationDistribution) = d.current_observation

function transition(maze::TMaze, s::TMazeState, a::Int64)
    d=create_transition_distribution(maze)
    d.reset = false
    # check if terminal
    if s.term
        # reset
        d.reset = true
        #copy!(d.current_state, s) # state doesn't change
        return d
    end
    # check if move into terminal move north or south
    if s.x == maze.n + 1
        if a == 1 || a == 3
            d.current_state = TMazeState(1,:none,true) # state now terminal
            return d
        elseif a == 4
            copy!(d.current_state, s)
            d.current_state.x -= 1
            return d
        else
            copy!(d.current_state, s)
            return d
        end
    end
    # check if move along hallway
    if a == 2
        copy!(d.current_state, s)
        d.current_state.x += 1
        return d
    end
    if a == 4
        copy!(d.current_state, s)
        s.x > 1 ? (d.current_state.x -= 1) : (nothing)
        return d
    end
    # if none of the above just stay in place
    copy!(d.current_state, s)
    return d
end

function reward(maze::TMaze, s::TMazeState, a::Int64)
    # check terminal
    s.term ? (return 0.0) : (nothing)
    # check if at junction
    if s.x == maze.n + 1
        # if at junction check action
        if (s.g == :north && a == 1) || (s.g == :south && a == 3)
            return 4.0
        elseif (s.g == :north && a == 3) || (s.g == :south && a == 1)
            return -0.1
        else
            return -0.1
        end
    end
    # if bump against wall
    if s.x < maze.n + 1 && (a == 1 || a == 3)
        return -0.1
    end
    return 0.0
end

# observation mapping
#    1      2        3         4         5
# goal N  goal S  corridor  junction  terminal
function observation(maze::TMaze, sp::TMazeState)
    d::TMazeObservationDistribution = create_observation_distribution(maze)
    sp.term ? (d.current_observation = 5; return d) : (nothing)
    x = sp.x; g = sp.g
    #if x == 1
    if x <= 2
        g == :north ? (d.current_observation = 1) : (d.current_observation = 2)
        return d
    end
    if 1 < x < (maze.n + 1)
        d.current_observation = 3
        return d
    end
    if x == maze.n + 1
        d.current_observation = 4
        return d
    end
    d.current_observation = 5
    return d
end

isterminal(m::TMaze, s::TMazeState) = s.term

discount(m::TMaze) = m.discount

function stateindex(maze::TMaze, s::TMazeState)
    s.term ? (return maze.n + 1) : (nothing)
    if s.g == :north
        return s.x + (s.x - 1)
    else
        return s.x + (s.x)
    end
end

function Base.convert(maze::TMaze, s::TMazeState)
    v = Array{Float64}(undef, 2)
    v[1] = s.x
    s.g == :north ? (v[2] = 0.0) : (v[2] = 1.0)
    return v
end

mutable struct MazeBelief
    last_obs::Int64
    mem::Symbol # memory
end
MazeBelief() = MazeBelief(1, :none)

mutable struct MazeUpdater <: Updater end
POMDPs.initialize_belief(bu::MazeUpdater, d::Any) = b

function POMDPs.update(bu::MazeUpdater, b::MazeBelief, a, o)
    bp::MazeBelief=create_belief(bu)
    bp.last_obs = o
    bp.mem = b.mem
    if o == 1
        bp.mem = :north
    end
    if o == 2
        bp.mem = :south
    end
    return bp
end



mutable struct MazeOptimal <: Policy end
POMDPs.updater(p::MazeOptimal) = MazeUpdater()

# 4 actions: go North, East, South, West (1, 2, 3, 4)
# observation mapping
#    1      2        3         4         5
# goal N  goal S  corridor  junction  terminal
function POMDPs.action(p::MazeOptimal, b::MazeBelief)
    # if don't know the goal go back
    if b.mem == :none
        return 4
    end
    if b.mem == :north && b.last_obs == 4
        return 1
    end
    if b.mem == :south && b.last_obs == 4
        return 3
    end
    return 2
end

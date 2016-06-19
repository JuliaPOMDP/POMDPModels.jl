type TMazeState
    x::Int64 # position in corridor
    g::Symbol # goal north or south
    term::Bool
end
TMazeState() = TMazeState(1, :north, false)
==(s1::TMazeState, s2::TMazeState) = s1.x == s2.x && s1.g == s2.g
hash(s::TMazeState, h::UInt64 = zero(UInt64)) = hash(s.x, hash(s.g, h))
function Base.copy!(s1::TMazeState, s2::TMazeState) 
    s1.x = s2.x
    s1.g = s2.g
    s1.term = s2.term
    return s1
end

type TMaze <: POMDP{TMazeState, Int64, Int64}
    n::Int64 # corridor length
    discount::Float64 # discount factor
    vec_state::Vector{Float64}
    vec_obs::Vector{Float64}
end
TMaze(n::Int64) = TMaze(n, 0.99, zeros(2), zeros(1)) 
TMaze() = TMaze(10)

n_states(m::TMaze) = 2 * (m.n + 1) + 1 # 2*(corr length + 1 (junction)) + 1 (term)
n_actions(::TMaze) = 4
n_observations(::TMaze) = 5

type TMazeStateSpace <: AbstractSpace{TMazeState}
    domain::Vector{TMazeState}
end
iterator(s::TMazeStateSpace) = s.domain
rand(rng::AbstractRNG, space::TMazeStateSpace, s::TMazeState) = space.domain[rand(rng, 1:length(space.domain))]
rand(rng::AbstractRNG, space::TMazeStateSpace) = space.domain[rand(rng, 1:length(space.domain))]

type TMazeSpace <: AbstractSpace{Int64}
    domain::Vector{Int64}
end
iterator(s::TMazeSpace) = s.domain
rand(rng::AbstractRNG, space::TMazeSpace, ao::Int64) = space.domain[rand(rng, 1:length(space.domain))]
rand(rng::AbstractRNG, space::TMazeSpace) = space.domain[rand(rng, 1:length(space.domain))]


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
    return TMazeStateSpace(space)
end
# 4 actions: go North, East, South, West (1, 2, 3, 4)
actions(maze::TMaze) = TMazeSpace(collect(1:4))
# 5 observations: 2 for goal (left or right) + 2 for in corridor or at intersection + 1 term
observations(maze::TMaze) = TMazeSpace(collect(1:5))

# transition distribution (actions are deterministic)
type TMazeStateDistribution <: AbstractDistribution{TMazeState}
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
iterator(d::TMazeStateDistribution) = reset ? (return [d.current_state]) : (return d.reset_states)

function pdf(d::TMazeStateDistribution, s::TMazeState) 
    if d.reset
        in(s, d.reset_states) ? (return 0.5) : (return 0.0)
    else
        s == d.current_state ? (return 1.0) : (return 0.0)
    end
end
function rand(rng::AbstractRNG, d::TMazeStateDistribution, s::TMazeState) 
    if d.reset 
        rand(rng) < 0.5 ? (copy!(s, d.reset_states[1])) : (copy!(s, d.reset_states[2]))
        return s
    else
        copy!(s, d.current_state)
        return s
    end
end
#rand(rng::AbstractRNG, d::TMazeStateDistribution) 

type TMazeInit <: AbstractDistribution{TMazeState}
    states::Vector{TMazeState}
    probs::Vector{Float64}
end
iterator(d::TMazeInit) = d.states
function initial_state_distribution(maze::TMaze)
    s = iterator(states(maze))
    ns = n_states(maze)
    p = zeros(ns) + 1.0 / (ns-1)
    p[end] = 0.0
    #s1 = TMazeState(1, :north, false)
    #s2 = TMazeState(1, :south, false)
    #d = TMazeInit([s1, s2])
    return TMazeInit(s, p)
end
function rand(rng::AbstractRNG, d::TMazeInit, s::TMazeState)
    #idx = nothing
    #rand(rng) < 0.5 ? (idx = 1) : (idx = 2)
    #copy!(s, d.states[idx])
    cat = Categorical(d.probs)
    idx = rand(cat)
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
type TMazeObservationDistribution <: AbstractDistribution{Int64}
    current_observation::Int64
end
create_observation_distribution(::TMaze) = TMazeObservationDistribution(1)
iterator(d::TMazeObservationDistribution) = [d.current_observation]

pdf(d::TMazeObservationDistribution, o::Int64) = o == d.current_observation ? (return 1.0) : (return 0.0)
rand(rng::AbstractRNG, d::TMazeObservationDistribution, o::Int64) = d.current_observation
rand(rng::AbstractRNG, d::TMazeObservationDistribution) = d.current_observation

function transition(maze::TMaze, s::TMazeState, a::Int64, 
                    d::TMazeStateDistribution=create_transition_distribution(maze))
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
function observation(maze::TMaze, a::Int64, sp::TMazeState, 
                     d::TMazeObservationDistribution = create_observation_distribution(maze))
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
function observation(maze::TMaze, s::TMazeState, a::Int64, sp::TMazeState,
                     d::TMazeObservationDistribution = create_observation_distribution(maze))
    return observation(maze, a, sp, d)
end

isterminal(m::TMaze, s::TMazeState) = s.term

discount(m::TMaze) = m.discount

create_state(::TMaze) = TMazeState()
create_action(::TMaze) = 1
create_observation(::TMaze) = 1

function state_index(maze::TMaze, s::TMazeState)
    s.term ? (return maze.n + 1) : (nothing)
    if s.g == :north
        return s.x + (s.x - 1)
    else
        return s.x + (s.x)
    end
end

function generate_o(maze::TMaze, s::TMazeState, rng::AbstractRNG, o::Int64=create_observation(maze))
    s.term ? (return 5) : (nothing)
    x = s.x; g = s.g
    #if x == 1
    if x <= 2
        g == :north ? (return 1) : (return 2) 
    end
    if 1 < x < (maze.n + 1)
        return 3
    end
    if x == maze.n + 1
        return 4
    end
    return 5
end

function vec(maze::TMaze, s::TMazeState)
    v = maze.vec_state
    v[1] = s.x
    s.g == :north ? (v[2] = 0.0) : (v[2] = 1.0)
    return v
end

function vec(maze::TMaze, o::Int64)
    maze.vec_obs[1] = o
    return maze.vec_obs
end

type MazeBelief 
    last_obs::Int64
    mem::Symbol # memory
end
MazeBelief() = MazeBelief(1, :none)

type MazeUpdater <: Updater{MazeBelief} end
POMDPs.create_belief(::MazeUpdater) = MazeBelief()
POMDPs.initialize_belief(bu::MazeUpdater, d::AbstractDistribution, b::MazeBelief=create_belief(bu)) = b

function POMDPs.update(bu::MazeUpdater, b::MazeBelief, a, o, bp::MazeBelief=create_belief(bu))
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



type MazeOptimal <: Policy{MazeBelief} end

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



type TMazeState
    x::Int64 # position in corridor
    g::Symbol # goal north or south
    term::Bool
end
TMazeState() = TMazeState(1, :north, false)
==(s1::TMazeState, s2::TMazeState) = s1.x == s2.x && s1.g == s2.g
hash(s::TMazeState, h::UInt64 = zero(UInt64)) = hash(s.x, hash(s.g, h))
function copy!(s1::TMazeState, s2::TMazeState) 
    s1.x = s2.x
    s1.g = s2.g
    s1.term = s2.term
    s1
end

type TMaze <: POMDP{TMazeState, Int64, Int64}
    n::Int64 # corridor length
    disocunt::Float64 # discount factor
end
TMaze(n::Int64) = TMaze(n, 0.95) 
TMaze() = TMaze(10)

n_states(m::TMaze) = 2 * (m.n + 1) + 1 # 2*(corr length + 1 (junction)) + 1 (term)
n_actions(::TMaze) = 4
n_observations(::TMaze) = 5

type TMazeStateSpace <: AbstractSpace{TMazeState}
    domain::Vector{TMazeState}
end
iterator(s::TMazeStateSpace) = s.domain
rand(rng::AbstractRNG, space::TMazeStateSpace, s::TMazeState) = space.domain[rand(rng, 1:length(space.domain))]

type TMazeSpace <: AbstractSpace{Int64}
    domain::Vector{Int64}
end
iterator(s::TMazeSpace) = s.domain
rand(rng::AbstractRNG, space::TMazeSpace, ao::Int64) = space.domain[rand(rng, 1:length(space.domain))]


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
end
create_transition_distribution(::TMaze) = TMazeStateDistribution(TMazeState())
iterator(d::TMazeStateDistribution) = [d.current_state]

pdf(d::TMazeStateDistribution, s::TMazeState) = s == d.current_state ? (return 1.0) : (return 0.0)
rand(rng::AbstractRNG, d::TMazeStateDistribution, s::TMazeState) = d.current_state
rand(rng::AbstractRNG, d::TMazeStateDistribution) = d.current_state

type TMazeInit <: AbstractDistribution{TMazeState}
    states::Vector{TMazeState}
end
function initial_state_distribution(maze::TMaze)
    s1 = TMazeState(1, :north, false)
    s2 = TMazeState(1, :south, false)
    d = TMazeInit([s1, s2])
    return d
end
function rand(rng::AbstractRNG, d::TMazeInit, s::TMazeState)
    idx = nothing
    rand(rng) < 0.5 ? (idx = 1) : (idx = 2)
    copy!(s, d.states[idx])
    return s
end
function pdf(d::TMazeInit, s::TMazeState)
    in(s, d) ? (return 0.5) : (return 0.0)
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
    # check if terminal
    if s.term
        copy!(d.current_state, s) # state doesn't change
        return d
    end
    # check if move into terminal move north or south
    if s.x == maze.n + 1 && (a == 1 || a == 3)
        d.current_state.term = true # state now terminal
        return d
    end
    # check if move along hallway
    if s.x < maze.n 
        if a == 2
            copy!(d.current_state, s)
            d.current_state.x += 1
        end
        if a == 4
            copy!(d.current_state, s)
            s.x > 1 ? (d.current_state.x -= 1) : (nothing)
        end
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
        else
            return -0.1
        end
    end
    # if bump against wall
    if s.x < maze.n + 1 && (a == 2 || a == 4)
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
    if x == 1
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


discount(m::TMaze) = m.disocunt

create_state(::TMaze) = TMazeState()
create_action(::TMaze) = 1
create_observation(::TMaze) = 1

function state_index(maze::TMaze, s::TMazeState)
    if s.g == :north
        return s.x
    else
        return s.x + (maze.n + 1)
    end
end

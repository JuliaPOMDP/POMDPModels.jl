@with_kw struct TMazeState
    x::Int64 = 1 # position in corridor
    g::Symbol = :north# goal north or south
end

@with_kw struct TMaze <: POMDP{Union{TMazeState,TerminalState}, Int64, Int64}
    n::Int64 = 10 # corridor length
    discount::Float64 = 0.99 # discount factor
end


# state space is (length of corr)*(north, south) + terminal
#                   |G|
# | | |x| | | | | | | |
#                   | |
function states(maze::TMaze)
    space = statetype(maze)[]
    for x in 1:(maze.n + 1), g in [:north, :south]
        push!(space, TMazeState(x, g))
    end
    push!(space, terminalstate) # terminal
    return space
end
stateindex(m::TMaze, s::TMazeState) = 2*s.x - (s.g==:north)
stateindex(m::TMaze, s::TerminalState) = 2*(m.n+1) + 1

# 4 actions: go North, East, South, West (1, 2, 3, 4)
actions(maze::TMaze) = 1:4
actionindex(maze::TMaze, i::Int) = i

# 5 observations: 2 for goal (left or right) + 2 for in corridor or at intersection + 1 term
observations(maze::TMaze) = 1:5
obsindex(maze::TMaze, i::Int) = i

function initialstate_distribution(maze::TMaze)
    s = states(maze)
    ns = length(s)
    p = zeros(ns) .+ 1.0 / (ns-1)
    p[end] = 0.0
    return SparseCat(s, p)
end

function transition(m::TMaze, s::TMazeState, a::Int64)
    if a == 1 || a == 3
        if s.x == m.n + 1
            Deterministic(terminalstate)
        else
            Deterministic(s)
        end
    elseif a == 2
        xp = min(s.x + 1, m.n + 1)
        return Deterministic(TMazeState(xp, s.g))
    elseif a == 4
        xp = max(s.x - 1, 1)
        return Deterministic(TMazeState(xp, s.g))
    end
end

transition(m::TMaze, s::TerminalState, a::Int64) = Deterministic(s)

function reward(m::TMaze, s::TMazeState, a::Int64)
    if s.x == m.n + 1
        # if at junction check action
        if (s.g == :north && a == 1) || (s.g == :south && a == 3)
            return 4.0
        else
            return -0.1
        end
    elseif a == 1 || a == 3
        # bump against wall
        return -0.1
    else
        return 0.0
    end
end

# observation mapping
#    1      2        3         4         5
# goal N  goal S  corridor  junction  terminal
function observation(m::TMaze, sp::TMazeState)
    if sp.x <= 2
        if sp.g == :north
            return Deterministic(1)
        else
            return Deterministic(2)
        end
    elseif sp.x == m.n+1
        return Deterministic(4)
    else
        return Deterministic(3)
    end
end

observation(m::TMaze, sp::TerminalState) = Deterministic(5)

discount(m::TMaze) = m.discount

function POMDPs.convert_s(::Type{A}, s::Union{TMazeState,TerminalState}, m::TMaze) where A <: AbstractArray
    return convert(A, [stateindex(m, s)])
end

# inverse of stateindex(m::TMaze, s::TMazeState) = 2*s.x - (s.g==:north)
function POMDPs.convert_s(::Type{S}, v::AbstractVector, m::TMaze) where S <: Union{TMazeState,TerminalState}
    i = first(v)
    if i == 2*(m.n + 1) + 1
        return terminalstate
    end

    if i%2 == 0
        g = :south
    else
        g = :north
    end
    x = div(i-1, 2) + 1
    @assert x <= m.n + 1
    return TMazeState(x, g)
end


struct MazeBelief
    last_obs::Int64
    mem::Symbol # memory
end
MazeBelief() = MazeBelief(1, :none)

struct MazeUpdater <: Updater end
POMDPs.initialize_belief(bu::MazeUpdater, d::Any) = d

function POMDPs.update(bu::MazeUpdater, b::MazeBelief, a, o)
    mem = b.mem
    if o == 1
        mem = :north
    end
    if o == 2
        mem = :south
    end
    return MazeBelief(o, mem)
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

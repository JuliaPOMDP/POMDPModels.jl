#################################################################
# This file implements the grid world problem as an MDP.
# In the problem, the agent is tasked with navigating in a
# stochatic environemnt. For example, when the agent chooses
# to go right, it may not always go right, but may go up, down
# or left with some probability. The agent's goal is to reach the
# reward states. The states with a positive reward are terminal,
# while the states with a negative reward are not.
#################################################################

using POMDPDistributions

#################################################################
# States and Actions
#################################################################
# state of the agent in grid world
type GridWorldState # this is not immutable because of how it is used in transition(), but maybe it should be
	x::Int64 # x position
	y::Int64 # y position
    bumped::Bool # bumped the wall or not in previous step
    done::Bool # entered the terminal reward state in previous step
    GridWorldState(x,y,bumped,done) = new(x,y,bumped,done)
    GridWorldState() = new()
end
# simpler constructors
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false,false)
# for state comparison
==(s1::GridWorldState,s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y && s1.bumped == s2.bumped && s1.done == s2.done
# for hashing states in dictionaries in Monte Carlo Tree Search
posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y
hash(s::GridWorldState, h::UInt64 = zero(UInt64)) = hash(s.x, hash(s.y, hash(s.bumped, hash(s.done, h))))
Base.copy!(dest::GridWorldState, src::GridWorldState) = (dest.x=src.x; dest.y=src.y; dest.bumped=src.bumped; dest.done=src.done; return dest)

# action taken by the agent indeicates desired travel direction
immutable GridWorldAction
    direction::Symbol
    GridWorldAction(d) = new(d)
    GridWorldAction() = new()
end 
==(u::GridWorldAction, v::GridWorldAction) = u.direction == v.direction
hash(a::GridWorldAction, h::UInt) = hash(a.direction, h)

#################################################################
# Grid World MDP
#################################################################
# the grid world mdp type
type GridWorld <: MDP{GridWorldState, GridWorldAction}
	size_x::Int64 # x size of the grid
	size_y::Int64 # y size of the grid
	reward_states::Vector{GridWorldState} # the states in which agent recieves reward
	reward_values::Vector{Float64} # reward values for those states
    bounds_penalty::Float64 # penalty for bumping the wall
    tprob::Float64 # probability of transitioning to the desired state
    terminals::Set{GridWorldState}
    discount_factor::Float64 # disocunt factor
end
# we use key worded arguments so we can change any of the values we pass in 
function GridWorld(;sx::Int64=10, # size_x
                    sy::Int64=10, # size_y
                    rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)],
                    rv::Vector{Float64}=[-10.,-5,10,3], 
                    penalty::Float64=-1.0, # bounds penalty
                    tp::Float64=0.7, # tprob
                    discount_factor::Float64=0.95)
    terminals = Set{GridWorldState}()
    for (i,v) in enumerate(rv)
        if v > 0.0
            push!(terminals, rs[i])
        end
    end
    return GridWorld(sx, sy, rs, rv, penalty, tp, terminals, discount_factor)
end

create_state(::GridWorld) = GridWorldState()
create_action(::GridWorld) = GridWorldAction()

#################################################################
# State and Action Spaces
#################################################################
# state space
type GridWorldStateSpace <: AbstractSpace
    states::Vector{GridWorldState}
end
# action space
type GridWorldActionSpace <: AbstractSpace
    actions::Vector{GridWorldAction}
end
# returns the state space
function states(mdp::GridWorld)
	s = GridWorldState[] 
	size_x = mdp.size_x
	size_y = mdp.size_y
    for d = 0:1, b = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
        push!(s, GridWorldState(x,y,b,d))
    end
    return GridWorldStateSpace(s)
end
# returns the action space
function actions(mdp::GridWorld, s=nothing)
	acts = [GridWorldAction(:up), GridWorldAction(:down), GridWorldAction(:left), GridWorldAction(:right)]
	return GridWorldActionSpace(acts)
end
POMDPs.actions(mdp::GridWorld, s::GridWorldState, as::GridWorldActionSpace) = as;

# returns an iterator over states or action (arrays in this case)
iterator(space::GridWorldStateSpace) = space.states
iterator(space::GridWorldActionSpace) = space.actions

# sampling and mutating methods
rand(rng::AbstractRNG, space::GridWorldStateSpace, s::GridWorldState=GridWorldState(0,0)) = space.states[rand(rng, 1:end)]
rand(space::GridWorldStateSpace) = space.states[rand(1:end)]

rand(rng::AbstractRNG, space::GridWorldActionSpace, a::GridWorldAction=GridWorldAction(:up)) = space.actions[rand(rng,1:end)]
rand(space::GridWorldActionSpace) = space.actions[rand(1:end)]

#################################################################
# Distributions
#################################################################

type GridWorldDistribution <: AbstractDistribution
    neighbors::Array{GridWorldState}
    probs::Array{Float64} 
    cat::Categorical
end

function create_transition_distribution(mdp::GridWorld)
    # can have at most five neighbors in grid world
    neighbors =  [GridWorldState(i,i) for i = 1:5]
    probabilities = zeros(5) + 1.0/5.0
    cat = Categorical(5)
    return GridWorldDistribution(neighbors, probabilities, cat)
end

# returns an iterator over the distirubtion
function POMDPs.iterator(d::GridWorldDistribution)
    return d.neighbors
end

function pdf(d::GridWorldDistribution, s::GridWorldState)
    for (i, sp) in enumerate(d.neighbors)
        if s == sp
            return d.probs[i]
        end
    end   
    return 0.0
end

# TODO these should be cleaned up once rand() stabilizes in pomdps
function rand(rng::AbstractRNG, d::GridWorldDistribution, s::GridWorldState=GridWorldState(0,0))
    set_prob!(d.cat, d.probs) # fill the Categorical distribution with our state probabilities
    d.neighbors[rand(rng, d.cat)] # sample a neighbor state according to the distribution c
end
#= # Don't need these, right?
function rand(rng::AbstractRNG, d::GridWorldDistribution, s::GridWorldState)
    set_prob!(d.cat, d.probs) # fill the Categorical distribution with our state probabilities
    d.neighbors[rand(rng, d.cat)] # sample a neighbor state according to the distribution c
    copy!(s, sample)
end
function rand(rng::AbstractRNG, d::GridWorldDistribution)
    set_prob!(d.cat, d.probs) # fill the Categorical distribution with our state probabilities
    d.neighbors[rand(rng, d.cat)] # sample a neighbor state according to the distribution c
end
=#

n_states(mdp::GridWorld) = 4*mdp.size_x*mdp.size_y
n_actions(mdp::GridWorld) = 4

#check for reward state
function reward(mdp::GridWorld, state::GridWorldState, action::GridWorldAction, sp::GridWorldState)
    if state.done
        return 0.0
    end
	r = 0.0
	reward_states = mdp.reward_states
	reward_values = mdp.reward_values
	n = length(reward_states)
	for i = 1:n
		if posequal(state, reward_states[i]) 
			r += reward_values[i]
		end
	end 
    if state.bumped
        r += mdp.bounds_penalty
    end
	return r
end

#checking boundries- x,y --> points of current state
function inbounds(mdp::GridWorld,x::Int64,y::Int64)
	if 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y 
		return true 
	else 
		return false
	end
end

function inbounds(mdp::GridWorld,state::GridWorldState)
	x = state.x #point x of state
	y = state.y
	return inbounds(mdp, x, y)
end

function fill_probability!(p::Vector{Float64}, val::Float64, index::Int64)
	for i = 1:length(p)
		if i == index
			p[i] = val
		else
			p[i] = 0.0
		end
	end
end

#function transition!(d::GridWorldDistribution, mdp::GridWorld, state::GridWorldState, action::GridWorldAction)
function transition(mdp::GridWorld, state::GridWorldState, action::GridWorldAction, d::GridWorldDistribution)
	a = action.direction 
	x = state.x
	y = state.y 
    
    neighbors = d.neighbors
    probability = d.probs
    
    fill!(probability, 0.1)
    probability[5] = 0.0 

    neighbors[1].x = x+1; neighbors[1].y = y
    neighbors[2].x = x-1; neighbors[2].y = y
    neighbors[3].x = x; neighbors[3].y = y-1
    neighbors[4].x = x; neighbors[4].y = y+1
    neighbors[5].x = x; neighbors[5].y = y


    if state.done
        fill_probability!(probability, 1.0, 5)
        neighbors[5].done = true
        neighbors[5].bumped = state.bumped
        return d
    end

    for i = 1:5 neighbors[i].bumped = false end
    for i = 1:5 neighbors[i].done = false end 
    reward_states = mdp.reward_states
    reward_values = mdp.reward_values
	n = length(reward_states)
	for i = 1:n
		#if state == reward_states[i] && reward_values[i] > 0.0
		if posequal(state, reward_states[i]) && reward_values[i] > 0.0
			fill_probability!(probability, 1.0, 5)
            neighbors[5].done = true
            return d
		end
	end 

    if a == :right  
		if !inbounds(mdp, neighbors[1])
			fill_probability!(probability, 1.0, 5)
            neighbors[5].bumped = true
		else
			probability[1] = 0.7
		end

	elseif a == :left
		if !inbounds(mdp, neighbors[2])
			fill_probability!(probability, 1.0, 5)
            neighbors[5].bumped = true
		else
			probability[2] = 0.7
		end

	elseif a == :down
		if !inbounds(mdp, neighbors[3])
			fill_probability!(probability, 1.0, 5)
            neighbors[5].bumped = true
		else
			probability[3] = 0.7
		end

	elseif a == :up 
		if !inbounds(mdp, neighbors[4])
			fill_probability!(probability, 1.0, 5)
            neighbors[5].bumped = true
		else
			probability[4] = 0.7 
		end
	end

    count = 0
    new_probability = 0.1
    
    for i = 1:length(neighbors)
        if !inbounds(mdp, neighbors[i])
         count += 1
            probability[i] = 0.0
         end
     end
 
    if count == 1 
        new_probability = 0.15
    elseif count == 2
        new_probability = 0.3
    end 
    
    if count > 0 
        for i = 1:length(neighbors)
            if probability[i] == 0.1
               probability[i] = new_probability
            end
        end
    end
    d
end


function state_index(mdp::GridWorld, s::GridWorldState)
    return s2i(mdp, s)
end

function s2i(mdp::GridWorld, state::GridWorldState)
    sb = Int(state.bumped + 1)
    sd = Int(state.done + 1)
    return sub2ind((mdp.size_x, mdp.size_y, 2, 2), state.x, state.y, sb, sd)
end 


function isterminal(mdp::GridWorld, s::GridWorldState)
    s.done ? (return true) : (return false)
end

discount(mdp::GridWorld) = mdp.discount_factor

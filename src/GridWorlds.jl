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

# state of the agent in grid world
type GridWorldState
	x::Int64 # x position
	y::Int64 # y position
    bumped::Bool # bumped the wall or not in previous step
    done::Bool # entered the terminal reward state in previous step
end
# simpler constructor
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false,false)
# for state comparison
==(s1::GridWorldState,s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y

# the grid world mdp type
type GridWorld <: POMDP
	size_x::Int64 # x size of the grid
	size_y::Int64 # y size of the grid
	reward_states::Vector{GridWorldState} # the states in which agent recieves reward
	reward_values::Vector{Float64} # reward values for those states
    bounds_penalty::Float64 # penalty for bumping the wall
    discount_factor::Float64 # disocunt factor
end
function GridWorld(sx::Int64, sy::Int64; 
                   rs::Vector{GridWorldState}=GridWorldState[], rv::Vector{Float64}=Float64[],
                   penalty::Float64=-1.0, discount_factor::Float64=0.9)
    if isempty(rs)
        rs = [GridWorldState(5,5), GridWorldState(3,3), GridWorldState(2,2)]
        rv = [10,-10,5]
    end
    return GridWorld(sx, sy, rs, rv, penalty, discount_factor)
end

type GridWorldAction 
    direction::Symbol
end 

type GridWorldDistribution <: AbstractDistribution
    neighbors::Array{GridWorldState}
    probabilities::Array{Float64} 
    mdp::GridWorld
end
function rand(d::GridWorldDistribution) 
    c = Categorical(d.probabilities)
    return d.neighbors[rand(c)]
end
function rand!(s::GridWorldState, d::GridWorldDistribution)
    c = Categorical(d.probabilities)
    ns = d.neighbors[rand(c)]
    s.x = ns.x; s.y = ns.y; s.bumped = ns.bumped; s.done = ns.done
    s
end

create_state(mdp::GridWorld) = GridWorldState(1, 1)
create_action(mdp::GridWorld) = GridWorldAction(:up)

n_states(mdp::GridWorld) = 4*mdp.size_x*mdp.size_y
n_actions(mdp::GridWorld) = 4

Base.length(d::GridWorldDistribution) = length(d.neighbors)
function index(d::GridWorldDistribution, i::Int64)
    return s2i(d.mdp, d.neighbors[i]) 
end
function weight(d::GridWorldDistribution, i::Int64)
    return d.probabilities[i] 
end
function create_transition_distribution(mdp::GridWorld)
    neighbors =  [GridWorldState(1,1) , GridWorldState(1,1) , GridWorldState(1,1) ,GridWorldState(1,1),GridWorldState(1,1)] 
    probabilities = zeros(5) + 1/5.0
    return GridWorldDistribution(neighbors, probabilities, mdp) #d = create_transition(mdp)
end

#check for reward state
function reward(mdp::GridWorld, state::GridWorldState, action::GridWorldAction) #deleted action
    if state.done
        return 0.0
    end
	r = 0.0
	reward_states = mdp.reward_states
	reward_values = mdp.reward_values
	n = length(reward_states)
	for i = 1:n
		if state == reward_states[i]
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
    probability = d.probabilities 
    
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
		if state == reward_states[i] && reward_values[i] > 0.0
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


function index(mdp::GridWorld, s::GridWorldState)
    return s2i(mdp, s)
end

function s2i(mdp::GridWorld, state::GridWorldState)
    sb = int(state.bumped + 1)
    sd = int(state.done + 1)
    return sub2ind([mdp.size_x, mdp.size_y, 2, 2], [state.x, state.y, sb, sd])
end 



type StateSpace <: AbstractSpace
    states::Vector{GridWorldState}
end

type ActionSpace <: AbstractSpace
    actions::Vector{GridWorldAction}
end

domain(space::StateSpace) = space.states
domain(space::ActionSpace) = space.actions

rand(space::StateSpace) = space.states[rand(1:end)]
function rand!(state::GridWorldState, space::StateSpace)
    state = space.states[rand(1:end)]    
    state
end

rand(space::ActionSpace) = space.actions[rand(1:end)]
function rand!(action::GridWorldState, space::ActionSpace)
    action = space.actions[rand(1:end)]    
    action
end

function states(mdp::GridWorld)
	s = GridWorldState[] 
	size_x = mdp.size_x
	size_y = mdp.size_y
    #for x = 1:mdp.size_x, y = 1:mdp.size_y, b = 0:1, d = 0:1
    for d = 0:1, b = 0:1, y = 1:mdp.size_y, x = 1:mdp.size_x
        push!(s, GridWorldState(x,y,b,d))
    end
    return StateSpace(s)
end
states!(space::StateSpace, mdp::GridWorld, state::GridWorldState) = space

function actions(mdp::GridWorld)
	acts = [GridWorldAction(:up), GridWorldAction(:down), 
	GridWorldAction(:left), GridWorldAction(:right)]
	return ActionSpace(acts)
end
#function actions(a::ActionSpace, mdp::GridWorld, state::GridWorldState)
function actions(mdp::GridWorld, state::GridWorldState, a::ActionSpace=create_action(mdp))
    return a
end

discount(mdp::GridWorld) = mdp.discount_factor

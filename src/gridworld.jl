const GWPos = SVector{2,Int}

@with_kw struct SimpleGridWorld <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           = (10,10)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(9,3)=>10.0, GWPos(8,8)=>3.0)
    terminate_in::Set{GWPos}        = Set((GWPos(4,3), GWPos(4,6), GWPos(9,3), GWPos(8,8)))
    tprob::Float64                  = 0.7
    discount::Float64               = 0.95
end


# States

function POMDPs.states(mdp::SimpleGridWorld)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end
POMDPs.n_states(mdp::SimpleGridWorld) = prod(mdp.size) + 1
function POMDPs.stateindex(mdp::SimpleGridWorld, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(mdp.size)[s...]
    else
        return n_states(mdp)
    end
end

POMDPs.initialstate(mdp::SimpleGridWorld, rng::AbstractRNG) = GWPos(rand(rng, 1:mdp.size[1]), rand(rng, 1:mdp.size[2]))


# Actions

POMDPs.actions(mdp::SimpleGridWorld) = (:up, :down, :left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box
POMDPs.n_actions(mdp::SimpleGridWorld) = 4

const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4)

POMDPs.actionindex(mdp::SimpleGridWorld, a::Symbol) = aind[a]



# Transitions

POMDPs.isterminal(m::SimpleGridWorld, s::AbstractVector{Int}) = any(s.<0)


function POMDPs.transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol)
    if isterminal(mdp, s) || s in mdp.terminate_in
        return Deterministic(GWPos(-1,-1))
    end
    
    destinations = MVector{n_actions(mdp)+1, GWPos}(undef)
    destinations[1] = s

    # probs = MVector{n_actions(mdp)+1, Float64}() 
    probs = @MVector(zeros(n_actions(mdp)+1)) 
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(n_actions(mdp) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end

function inbounds(m::SimpleGridWorld, s::AbstractVector{Int})
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2]
end

# Rewards

POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)


# discount

POMDPs.discount(mdp::SimpleGridWorld) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end

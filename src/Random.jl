function RandomMDP(ns::Int64, na::Int64, discount::Float64)
    # random dynamics
    T = rand(ns, na, ns) 
    # normalize
    for i = 1:ns, j = 1:na
        T[:,j,i] /= sum(T[:,j,i])
    end
    # random rewards [-0.5, 0.5]
    R = rand(ns, na) - 0.5
    return DiscreteMDP(T, R, discount)
end
RandomMDP() = RandomMDP(100, 5, 0.9)


function RandomPOMDP(ns::Int64, na::Int64, no::Int64, discount::Float64)
    # random dynamics
    T = rand(ns, na, ns) 
    # random observation model
    O = rand(no, na, ns)
    # normalize
    for i = 1:ns, j = 1:na
        T[:,j,i] /= sum(T[:,j,i])
        O[:,j,i] /= sum(O[:,j,i])
    end
    # random rewards [-0.5, 0.5]
    R = rand(ns, na) - 0.5
    return DiscretePOMDP(T, R, O, discount)
end
RandomPOMDP() = RandomPOMDP(100, 5, 20, 0.9)

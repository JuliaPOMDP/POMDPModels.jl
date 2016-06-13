using POMDPModels

R = [-1. -100 10; -1 10 -100]

T = zeros(2,3,2)
T[:,:,1] = [1. 0.5 0.5; 0 0.5 0.5]
T[:,:,2] = [0. 0.5 0.5; 1 0.5 0.5]

O = zeros(2,3,2)
O[:,:,1] = [0.85 0.5 0.5; 0.15 0.5 0.5] 
O[:,:,2] = [0.15 0.5 0.5; 0.85 0.5 0.5] 

pomdp1 = TigerPOMDP()

pomdp2 = DiscretePOMDP(T, R, O, 0.95)

probability_check(pomdp1)
probability_check(pomdp2)

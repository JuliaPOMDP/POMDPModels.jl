# POMDPModels
[![Build Status](https://travis-ci.org/JuliaPOMDP/POMDPModels.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/POMDPModels.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/POMDPModels.jl/badge.svg)](https://coveralls.io/github/JuliaPOMDP/POMDPModels.jl)

This package provides the following models for POMDPs.jl:

* [SimpleGridWorld](src/gridworld.jl)
* Tiger
* Crying Baby
* Random 
* Mountain Car
* Inverted Pendulum
* T-Maze

## Usage

To use POMDPModels, simply load it and initialize a model. Note: to interact with the models using the POMDPs.jl interface, you must also import POMDPs. The model supports the basic functions required by many of the JuliaPOMDP solvers. For example:

```julia
using POMDPs
using POMDPModels

pomdp = TigerPOMDP()
# do what you would do with a POMDP model, for example use QMDP to solve it
using QMDP
solver = QMDPSolver()
policy = solve(solver, pomdp) # compute a pomdp policy
```

You can initialize the other pomdp types in the module in the following way:
```julia
using POMDPModels

pomdp = TigerPOMDP()
pomdp = BabyPOMDP()
pomdp = RandomPOMDP()

mdp = GridWorld()
mdp = RandomMDP()
```

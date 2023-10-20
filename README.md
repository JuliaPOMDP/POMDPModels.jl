# POMDPModels
[![CI](https://github.com/JuliaPOMDP/POMDPModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/POMDPModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaPOMDP/POMDPModels.jl/branch/master/graph/badge.svg?token=xPLiTP3IVt)](https://codecov.io/gh/JuliaPOMDP/POMDPModels.jl)

This package provides the following models for POMDPs.jl:

* [SimpleGridWorld](src/gridworld.jl)
* [Tiger](src/TigerPOMDPs.jl)
* [Crying Baby](src/CryingBabies.jl)
* [Random](src/Random.jl)
* [Mountain Car](src/MountainCar.jl)
* [Inverted Pendulum](src/InvertedPendulum.jl)
* [T-Maze](src/TMazes.jl)
* [MiniHallway](src/MiniHallway.jl)
* [LightDark](src/LightDark.jl)
* [Tabular](src/Tabular.jl)

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

mdp = SimpleGridWorld()
mdp = RandomMDP()
```

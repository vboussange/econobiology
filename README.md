# Econobiology

This repository contains the code used for the paper 

> *Processes analogous to ecological interactions and dispersal shape the dynamics of economic activities*, Boussange, V., Sornette, D., Lischke, H., Pellissier, L. (2023).

More specifically:
- `MiniBatchInference.jl/` contains the source code for the inference utilities. It corresponds to a legacy version of [PiecewiseInference.jl](https://github.com/vboussange/PiecewiseInference.jl).
- `Econobio.jl/` contains generic utility functions to pre- and post-process input data, and implements the dynamic community models.
- `code/` contains all scripts related to the specific simulation runs
- `figure/` contains all scripts to generate the manuscript figures and crunch the raw simulation results.

All scripts are written in the Julia programming language. A short description of the purpose of each script is placed in each script preamble. The scripts can be executed out of the box by activating the environment stored in the `Project.toml` and `Manifest.toml` files in the root folder.

To activate the environment in an interactive session, type in the Julia REPL

```julia
julia>] activate .
julia>] instantiate
```
To simply run a script, type in the terminal
```
> julia --project=. name_of_the_script.jl
```
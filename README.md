# Directories
* [pacsim](https://github.com/PacSim/pacsim), used for vehicle model etc. python bindings are built in Dockerfile
* pipeline: where the magic happens, check [README](pipeline/README.md)
* configs: just some generic configs
* tracks: track files (PacSim format)

# Prerequisites
* Nvidia GPU (for plug-and-play. gpu flag can be removed from docker-compose, the state-based-model is trained on cpu)
* Docker, docker-compose and nvidia-container-toolkit (when running with GPU)

# Pre trained weights

This repo includes the weights of the best state based and vision based model. If you want to run them, copy from weights to pipeline/networks

# How to Run

Pull, initialize git submodule if not done already (git submodule update --init --recursvie)

1. VSCode Devcontainer
    * Press f1
    * Rebuild and reopen in container
    
2. Docker compose
    * docker compose up
    * docker exec -it fs-rl bash

* cd into /root/workspace/pipeline and check out [README](pipeline/README.md)

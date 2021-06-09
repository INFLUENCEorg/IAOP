# Influence-Augmented Online Planning for Complex Environments

This is the codebase accompanying the NeurIPS2020 paper "[Influence-Augmented Online Planning for Complex Environments](https://proceedings.neurips.cc//paper/2020/hash/2e6d9c6052e99fcdfa61d9b9da273ca2-Abstract.html)" by Jinke He, Miguel Suau and Frans A. Oliehoek.

Disclaimer: the codebase has only been tested on Ubuntu and Arch Linux. 

**Update**: We fixed a few minor bugs found in the codebase and updated part of the results in the [arxiv paper](https://arxiv.org/abs/2010.11038).

## Dependencies

### For online planning experiments:
* [Singularity](https://sylabs.io/docs/)
* [libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip) (C++ version of PyTorch): `download and unzip libtorch into third-party/libtorch`
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): `git clone https://github.com/jbeder/yaml-cpp third-party/yaml-cpp`

### For training the influence predictors and plotting the results:
* python3
* pytorch
* matplotlib
* jupyter notebook

## Training the influence predictors
* instructions for training the influence predictors will be added soon.
* pretrained influence predictors are available under [models/](models).

## Online Planning Experiments

### Singularity Container
We implemented our online planning experiments in C++ and provided a Singularity definition ([singularity/FADMEN.def](singularity/FADMEN.def)) file to resolve the dependencies. 

To run the code, first build the singularity container with the command: `sudo singularity build singularity/FADMEN.sif singularity/FADMEN.def`.

To execute a command under the singularity container, use `./run` + command.

### Compile
`./run bash scripts/build.sh`

### Reproducing results

#### General
`./run ./scripts/run_benchmark` + path to config file   
for example, `./scripts/run_benchmark configs/GAC/5agent/global.yaml`

#### Grab A Chair - num-agents: Figure 2(a) and 2(b)
`./run ./scripts/run_benchmark configs/GAC/{X}agent/{Y}.yaml`     
for X in `num_agents=[5,9,17,33,65,129]` and    
for Y in `algos=["global", "inf_gru_H8_D1000", "inf_rand"]` 

#### Grab A Chair - coupling: Figure 3
`./run ./scripts/run_benchmark configs/GAC-coupling/p{X}/{Y}.yaml`     
for X in `[01,02,03,04,05,06,07,08,09]` and    
for Y in `algos=["global", "inf_gru_H8_D1000SPLIT[0-3]", "inf_rand"]` 

#### Grid Traffic Control - real-time: 5(b), 5(c) and 5(d)
`./run ./scripts/run_benchmark configs/GTC/{X}sec/{Y}.yaml`     
for X in `seconds=[1,2,4,8,16,32,48,64]` and    
for Y in `algos=["global", "inf_rnn_H2_D1000SPLIT[0-*]", "inf_rand"]` 

### Plotting results
see [notebooks/Plot-Results.ipynb](notebooks/Plot-Results.ipynb)

## Acknowledgment
This project had received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 758824 — INFLUENCE).

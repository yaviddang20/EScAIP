# EScAIP: Efficiently Scaled Attention Interatomic Potential

This repository contains the WIP test version of the Efficiently Scaled Attention Interatomic Potential.

## Install

Step 1: Install mamba solver for conda (optional)

```bash
conda install mamba -n base -c conda-forge
```

Step 2: Check the CUDA is in `PATH` and `LD_LIBRARY_PATH`

```bash
$ echo $PATH | tr ':' '\n' | grep cuda
/usr/local/cuda/bin

$ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
/usr/local/cuda/lib64
```

If not, add something like following (depends on the location) to your `.bashrc` or `.zshrc`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Step 3: Install the dependencies

```bash
mamba env create -f env.yml
conda activate escaip
```

Step 4: Install FairChem core package

```bash
git submodule update --init --recursive
pip install -e fairchem/packages/fairchem-core
```

Step 4: Install the pre-commit (if you want to contribute)

```bash
pre-commit install
```

## Train

Use `main.py` to train the models. Th usage is similar to FairChem (refer to the [FairChem documentation](https://fair-chem.github.io/core/model_training.html) for more details):

Single GPU training:

```bash
python main.py --mode train --config-yml {CONFIG} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

Use `start_exp.py` to start a run in the background:

```bash
python start_exp.py --config-yml {CONFIG} --cvd {GPU NUM} --run-dir {RUNDIR} --timestamp-id {TIMESTAMP} --checkpoint {CHECKPOINT}
```

Multi-GPU training (same device):

```bash
torchrun --standalone --nproc_per_node={N} main.py --distributed --num-gpus {N} {...}
```

For distributed training on NERSC, refer to the [NERSC distributed training with submitit](NERSC_dist_train.md).

## Simulation

Use `simulate.py` to simulate the models. It requires to install the MDSim package:

```bash
pip install -e MDsim
```

A separate config file is needed for the simulation. Some example configs can be found in `configs/s2ef/MD22/datasets/{DATASET}/simulation.yml`.

```bash
python simulate.py --simulation_config_yml {SIMULATION CONFIG} --model_dir {CHECKPOINT DIR} --model_config_yml {MODEL CONFIG}--identifier {IDENTIFIER}
```

### Example Usage

For example, to simulate on MD22 DHA split:

```bash
python simulate.py --simulation_config_yml configs/s2ef/MD22/datasets/DHA/simulation.yml --model_dir checkpoints/MD22_DHA/ --model_config_yml configs/s2ef/MD22/EScAIP/DHA.yml --identifier test_simulation
```

After the simulation, the results can be found in the checkpoint directory under the `identifier` folder. Use `scripts/analyze_rollouts_md17_22.py` to analyze the results:

```bash
PYTHONPATH=./ python scripts/analyze_rollouts_md17_22.py --md_dir checkpoints/MD22_DHA/md_sim_test_simulation --gt_traj /data/md22/md22_AT-AT.npz --xlim 25
```

Refer to the [MDSim repo](https://github.com/kyonofx/MDsim) for more details.


## Model Architecture

Refer to the [model architecture](model_architecture.md) for more details. A detailed description of model configurations can be found [here](configs/example_config_EScAIP.yml).

Some notes on the configs:
- The model uses `torch.compile` to compile the model for better performance. [But it's not supporting second order gradient.](https://github.com/pytorch/pytorch/issues/91469) So to use gradient energy, it has to be disabled.
    - The implicit batch needs to be padded to use `torch.compile`. The size is controlled by `max_num_nodes_per_batch`.
- The attention kernels are selected by `atten_name`, which has 4 options:
    - `math`: the default attention by pytorch, supports all datatypes and second order gradient.
    - `memory_efficient`: the memory efficient kernel by pytorch, supports fp32 and fp16, but no second order gradient.
    - `flash`: the flash attention kernel by pytorch, supports only fp16, no second order gradient.
    - `xformers`: the xformers attention kernel, supports fp32 and fp16, no second order gradient. It's also not working with `torch.compile`.
- The `use_fp16_backbone` will turn the graph attention backbone to fp16, and the output head will still be fp32. For now, we're opting to use AutoMixedPrecision instead, as it's more stable.

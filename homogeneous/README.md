# Netherwind Air Combat MARL Benchmark

> **This repository is a self-contained submodule of the Netherwind Air
> Combat Simulation Platform** — specifically its pure-Python multi-agent
> reinforcement learning training and 3-D visualization scenario module.
> The full Netherwind platform is a considerably larger simulation system;
> what is released here is exactly the scenario layer on which every result
> in the accompanying paper was trained and evaluated. It runs standalone —
> no other platform modules are required — and it intentionally contains no
> content from the rest of the platform.

A physics-based multi-agent reinforcement learning benchmark for cooperative
air combat, built on the Netherwind Air Combat Simulation Platform
(JSBSim F-16-class 6-DOF flight dynamics), together with the **BCA**
(Bilateral-value Curriculum Agent) algorithm and seven MARL baselines.

## Installation

```bash
pip install -r requirements.txt
```

**Python 3.12 on Windows x64 is required.** The simulation core ships as
pre-compiled CPython-3.12 extension modules (`.pyd`); other interpreter
versions cannot load them. A C compiler is NOT required.

## Repository layout for researchers

| Path | Visibility | Contents |
|------|-----------|----------|
| `algorithms/` | **Source** | BCA (GRE + BVD + PBC) and baselines: IPPO, MAPPO, HAPPO, MADDPG, SAC, MAT, Random. Read, modify, retrain freely. |
| `train.py` | **Source** | Unified training entry point (all algorithms, all protocol flags). |
| `reeval_deterministic.py` | **Source** | Paired deterministic re-evaluation protocol (common random numbers, SWA checkpoints). |
| `visualizer.py` / `tactics_analyzer.py` | **Source** | Real-time 3-D battlefield renderer (driven by `train.py --visualize`) and trajectory analysis. |
| `docs/algorithm_interface.md` | Doc | How to write and register your own algorithm. |
| `reproduce_paper.md` | Doc | The exact command behind every number in the paper. |
| `env.pyd`, `missile.pyd`, `blue_policy.pyd`, `aircraft_models.pyd`, `jsbsim_adapter.pyd`, `jsbsim_aircraft.pyd`, `rewards/*.pyd` | **Binary** | The simulation core: environment dynamics, fire control, missile guidance, rule-based opponent, reward functions. Callable but not readable. |
| `jsbsim-master/` | Data | JSBSim aircraft/engine configuration data (public JSBSim content). |

The compiled simulation core exposes the exact same API as the training code
expects; `train.py` and `algorithms/` import it transparently. You can modify
any algorithm and retrain against the identical environment used in the paper.
A note on the protection model: Cython-compiled `.pyd` raises the bar for
casual inspection and modification of the combat logic; it is not a
cryptographic guarantee against a determined reverse-engineer.

## Quick start

Train BCA in the 3v3 scenario (100 episodes smoke test):

```bash
python launch.py train --algorithm bca --n_red 3 --n_blue 3 --episodes 100 \
    --lr 1e-4 --entropy 0.001 --blue_difficulty combat --reward_fn base \
    --eval_interval 25 --eval_episodes 20 --run_name demo
```

Train a baseline (identical protocol to the paper):

```bash
python launch.py train --algorithm ippo --n_red 3 --n_blue 3 --episodes 1000 \
    --lr 3e-4 --entropy 0.01 --blue_difficulty combat --reward_fn base \
    --run_name ippo_demo --seed 42
```

Visualize a policy in the real-time 3-D battlefield (the renderer is driven
through the training entry point; `visualizer.py` itself is a library):

```bash
python launch.py train --algorithm bca --n_red 3 --n_blue 3 --episodes 5 \
    --blue_difficulty combat --visualize --run_name viz_demo
```

## Reproducing the paper

See `reproduce_paper.md` for the exact command behind every number in the
paper, and `results/` for the evaluation CSVs those commands produce.

## License

This repository is released for academic use under the **MIT License** — see
`LICENSE`. The bundled `jsbsim-master/` data retains the JSBSim project's
original LGPL-2.1+ license, as noted in `LICENSE`.

If you use this benchmark in your research, please cite the accompanying
paper (citation entry to be added upon publication).

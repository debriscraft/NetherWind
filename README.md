# NetherWind Air-Combat MARL Framework (Unified Release)

One framework reproducing two companion studies from the Netherwind Air
Combat Simulation Platform, both with JSBSim six-degree-of-freedom flight
dynamics in the loop:

| Track | Paper | Setting | Algorithm |
|---|---|---|---|
| `homogeneous/` | Paper 1 | homogeneous multi-UAV formations (3v3 / 5v5) | **BCA** — Bilateral-value Curriculum Agent (graph-relational encoder + bilateral value decomposition + performance-based curriculum) |
| `heterogeneous/` | Paper 2 | heterogeneous fighter–UCAV formations (2v2, 3v2, 6v6) | **RAC-MAPPO** — role-attentive centralised MAPPO (declared role embeddings + set-attention critic + Pk-aware cue gate) |

## Intellectual-property notice

- **Public source**: the proposed algorithms and their building blocks —
  `homogeneous/algorithms/*.py`, `homogeneous/rewards/__init__.py`,
  `heterogeneous/sim/marl/rac_mappo.py`, plus all scenario/platform YAML
  configs and the thin `run_*.py` entry scripts.
- **Compiled binaries (`.pyd`)**: the simulation engine, trainers,
  evaluation machinery and visualizers ship as CPython-3.12 C extensions
  (Cython/MSVC build). They import and run like normal modules but their
  source is not distributed.
- **Python 3.12.x on 64-bit Windows is mandatory** — `.pyd` binaries are
  interpreter- and platform-specific.

## Requirements

```
Python 3.12.x (win_amd64)
pip install -r requirements.txt
```

Tested with: torch 2.6.0, numpy 2.4.4, jsbsim 1.3.1, gym-jsbsim, PyYAML,
matplotlib (see each track's `requirements.txt` for details).

## Quick start

### Homogeneous track (Paper 1, BCA)

```bash
cd homogeneous
python run_train.py --algorithm bca --n_red 3 --n_blue 3 --episodes 100 \
    --blue_difficulty combat --run_name demo --seed 42

# real-time 3-D battlefield visualization
python run_train.py --algorithm bca --n_red 3 --n_blue 3 --episodes 5 \
    --blue_difficulty combat --visualize --run_name viz_demo
```

See `homogeneous/reproduce_paper.md` for the exact command behind every
number in Paper 1.

### Heterogeneous track (Paper 2, RAC-MAPPO)

```bash
cd heterogeneous
python tests/smoke_bridge.py        # JSBSim bridge smoke test (~1 min)
python run_phase6_train.py          # main training: RAC-MAPPO + baselines
python run_ladder_eval.py           # 100-episode paired-seed ladder evaluation
python run_ablations.py             # ablation variants
python run_selfplay_eval.py         # self-play control
python run_scale_train.py           # 3-vs-2 / 6-vs-6 scalability
python run_pm_seeds.py              # point-mass fidelity control
```

See `heterogeneous/reproduce_paper2.md`.

## Datasets

The complete training and evaluation logs behind every table and figure of
both papers (data only — no figures, no model weights) are published
alongside this framework. Paper 1's dataset is on ScienceDB
(https://doi.org/10.57760/sciencedb.46055); Paper 2's dataset ships as the
companion `RACMAPPO_Dataset` package.

## Citation

If you use this framework, please cite both companion papers (BibTeX entries
will be added upon publication).

## License

See `LICENSE` (homogeneous track). Academic research use.

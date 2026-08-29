# NetherWind Air-Combat MARL Framework (Unified Release)

One framework reproducing three companion studies from the Netherwind Air
Combat Simulation Platform, all with JSBSim six-degree-of-freedom flight
dynamics in the loop:

| Track | Paper | Setting | Algorithm |
|---|---|---|---|
| `homogeneous/` | Paper 1 | homogeneous multi-UAV formations (3v3 / 5v5) | **BCA** — Bilateral-value Curriculum Agent (graph-relational encoder + bilateral value decomposition + performance-based curriculum) |
| `heterogeneous/` | Paper 2 | heterogeneous fighter–UCAV formations (2v2, 3v2, 6v6) | **RAC-MAPPO** — role-attentive centralised MAPPO (declared role embeddings + set-attention critic + Pk-aware cue gate) |
| `adaptive/` | Paper 3 | variable-scale formations, one policy for 2v2–6v6, zero-shot to 10v10/20v20 | **ANA-MAPPO** — agent-number-adaptive MAPPO (entity tokenizer + permutation-equivariant set-attention actor/critic, cardinality-independent) |

## Intellectual-property notice

- **Public source**: the proposed algorithms and their building blocks —
  `homogeneous/algorithms/*.py`, `homogeneous/rewards/__init__.py`,
  `heterogeneous/sim/marl/rac_mappo.py`,
  `adaptive/sim/marl/ana_mappo.py`, `adaptive/sim/envs/entity_tokens.py`,
  plus all scenario/platform YAML configs.
- **Compiled binaries**: the simulation engine, trainers, evaluation
  machinery and visualizers ship as compiled artifacts — CPython-3.12 C
  extensions (`.pyd`, Cython/MSVC build) in the `homogeneous/` and
  `heterogeneous/` tracks, and CPython-3.12 bytecode (`.pyc`) in the
  `adaptive/` track. They import and run like normal modules but their
  source is not distributed.
- **Python 3.12.x on 64-bit Windows is mandatory** — the compiled binaries
  are interpreter- and platform-specific.

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
python launch.py train --algorithm bca --n_red 3 --n_blue 3 --episodes 100 \
    --blue_difficulty combat --run_name demo --seed 42

# real-time 3-D battlefield visualization
python launch.py train --algorithm bca --n_red 3 --n_blue 3 --episodes 5 \
    --blue_difficulty combat --visualize --run_name viz_demo
```

See `homogeneous/reproduce_paper.md` for the exact command behind every
number in Paper 1.

### Heterogeneous track (Paper 2, RAC-MAPPO)

```bash
cd heterogeneous
python launch.py tests.smoke_bridge        # JSBSim bridge smoke test (~1 min)
python launch.py run_phase6_train          # main training: RAC-MAPPO + baselines
python launch.py run_ladder_eval           # 100-episode paired-seed ladder evaluation
python launch.py run_ablations             # ablation variants
python launch.py run_selfplay_eval         # self-play control
python launch.py run_scale_train           # 3-vs-2 / 6-vs-6 scalability
python launch.py run_pm_seeds              # point-mass fidelity control
```

See `heterogeneous/reproduce_paper2.md`.

### Adaptive track (Paper 3, ANA-MAPPO)

```bash
cd adaptive
python tests/test_entity_tokens.pyc     # tokenizer & environment sanity checks
python tests/test_ana_network.pyc       # symmetry tests: equivariance, invariance,
                                        # cardinality independence (598,957 params)
python smoke_env.pyc                    # one short environment rollout
python run_gated.pyc                    # main training (gated two-phase curriculum)
python eval_paired.pyc                  # paired evaluation vs opponent pool
python eval_zero_shot.pyc               # zero-shot transfer across team sizes
python eval_extrapolation.pyc           # extrapolation to 10v10 / 20v20
python eval_baseline.pyc                # MAPPO / QMIX baselines
python compute_pvalues.pyc              # Wilcoxon signed-rank p-values
```

See `adaptive/README.md`.

## Datasets

The complete training and evaluation logs behind every table and figure of
the three papers (data only — no figures, no model weights) are published
alongside this framework. Paper 1's dataset is on ScienceDB
(https://doi.org/10.57760/sciencedb.46055); Paper 2's dataset is on ScienceDB
(https://doi.org/10.57760/sciencedb.00zq1); Paper 3's dataset is on ScienceDB
(https://doi.org/10.57760/sciencedb.010uw).

## Citation

If you use this framework, please cite the companion papers (BibTeX entries
will be added upon publication).

## License

See `LICENSE` (homogeneous track). Academic research use.

# CA-MAPPO — Code Release (paper08)

Minimal, reproducible framework for the paper
**"Cardinality-Adaptive Multi-Agent Reinforcement Learning for
Variable-Scale Cooperative Aerial Games"**.

This package lets a reviewer or reader **run** every experiment pipeline that
produced the paper's numbers, while only the core contribution — the proposed
algorithm and its entity tokenizer — is distributed as readable source.

> Naming note: the paper names the method **CA-MAPPO** (Cardinality-Adaptive
> MAPPO). The implementation keeps the original module/class identifiers
> (`sim/marl/ana_mappo.py`, class `ANAMAPPO`) for compatibility with the
> released logs; they are the same algorithm.

## What is public vs. compiled

| Component | Path | Form |
|---|---|---|
| CA-MAPPO algorithm (set-attention actor/critic) | `sim/marl/ana_mappo.py` | **public source (.py)** |
| Entity tokenizer (per-agent/per-opponent tokens) | `sim/envs/entity_tokens.py` | **public source (.py)** |
| Entry scripts (train/eval/record/analysis) | `*.pyc` (top level) | compiled, runnable |
| Trainers (CA-MAPPO / MAPPO / QMIX), baselines | `train/*.pyc` | compiled |
| Environment, dynamics, missile, visualization | `sim/**/*.pyc` | compiled |
| Unit tests (symmetry & tokenizer) | `tests/*.pyc` | compiled, runnable |
| JSBSim flight-dynamics engine (upstream) | `jsbsim-master/` | upstream project |
| Configs | `configs/` | plain text / yaml |

Only the two algorithm files at the heart of the contribution are readable;
everything else ships as CPython 3.12 bytecode and can be executed but not
inspected.

## Requirements

- Python **3.12** (the `.pyc` files are CPython 3.12 bytecode; other versions
  cannot load them)
- `pip install -r requirements.txt`
- A C++ toolchain is **not** required; JSBSim is used through its Python
  bindings already present in the environment used for the paper.

## Quick start (smoke test, ~1 min)

```bash
# environment + tokenizer sanity checks
python tests/test_entity_tokens.pyc

# network symmetry tests: permutation equivariance/invariance,
# cardinality independence (598,957 actor params, independent of n)
python tests/test_ana_network.pyc

# one short environment rollout
python smoke_env.pyc
```

All tests should print `PASSED`.

## Reproducing the paper

All commands below are the exact pipelines used for the paper's tables and
figures. Random seeds are fixed in the configs; results land under `runs/`.

```bash
# --- main training (gated two-phase curriculum, nv2..nv6 mixed) ---
python run_gated.pyc                       # full CA-MAPPO training

# --- evaluation protocols ---
python eval_paired.pyc                     # paired evaluation vs opponent pool
python eval_by_level.pyc                   # per-opponent-level win rates
python eval_zero_shot.pyc                  # zero-shot transfer across team sizes
python eval_extrapolation.pyc              # extrapolation to 8v8-20v20 (K_max lifted)
python eval_kmax_ablation.pyc              # token-budget (K_max=14) ablation, Table 12 last column
python eval_baseline.pyc                   # MAPPO / QMIX baselines

# --- per-episode analysis used in the paper ---
python record_episodes.pyc                 # record full engagement episodes
python analyze_tactics.pyc                 # emergent-tactic analysis
python compute_pvalues.pyc                 # Wilcoxon signed-rank p-values
```

Each entry script prints its own progress and writes JSON/JSONL/NPZ logs in
the same layout as the released dataset (`CAMAPPO_Dataset`), so released
numbers can be regenerated and diffed directly.

## Dataset

The companion data release (`CAMAPPO_Dataset/`) contains the raw training
and evaluation logs (JSON/JSONL/NPZ) for every table and figure — no model
weights, no figures. See the dataset's own README for the file map.

## Citation

If this code is used, please cite the paper; the companion dataset is
deposited on figshare at https://doi.org/10.6084/m9.figshare.33424381 .

## License

For academic research use. The JSBSim engine under `jsbsim-master/` retains
its own upstream license (LGPL).

# Reproducing Paper 2 (RAC-MAPPO, heterogeneous fighter–UCAV formations)

Every number in the paper traces to the commands below. All runs use the
public entry scripts; the engine and trainers are compiled `.pyd` modules.
Hardware reference: a full main-matrix run (RAC-MAPPO + baselines, 3 seeds)
takes on the order of hours on a desktop CPU + GPU.

## 1. Main comparison matrix (Tables: win rate / kill ratio / survival)

Train RAC-MAPPO and the no-coordination-MAPPO and IPPO baselines under the
identical protocol, then evaluate on the four-level scripted adversary
ladder with paired seeds:

```bash
python launch.py run_phase6_train      # trains rac_mappo, b1_mappo_nocoord, b3_ippo
python launch.py run_ladder_eval       # 100 paired episodes per algorithm x level cell
```

Outputs land in `runs/<run_name>/` (`training_log.jsonl`, `eval.jsonl`,
`episodes.jsonl`, checkpoints) and the evaluation summary under
`runs/eval_final__*/`.

## 2. Ablation study (team reward / curriculum / attention critic /
   role embedding / parameter sharing / Pk gate)

```bash
python launch.py run_ablations
```

Each variant is retrained from scratch under the identical protocol and
evaluated on 100 paired episodes against L2/L3.

## 3. Self-play control (Section: self-play)

```bash
python launch.py run_selfplay_eval
```

Runs the mirror learned-vs-learned control (100 episodes) underlying the
doctrine-validation argument.

## 4. Scalability (3-vs-2 and 6-vs-6)

```bash
python launch.py run_scale_train
```

Zero-shot transfer, fine-tuning and from-scratch conditions per scale.

## 5. Fidelity control (point-mass vs JSBSim 6-DoF)

```bash
python launch.py run_pm_seeds
```

Trains and evaluates the point-mass control group behind the fidelity
comparison (cross-seed variance inflation).

## Notes

- The proposed algorithm is open source: `sim/marl/rac_mappo.py`.
- All other modules are compiled `.pyd` binaries (CPython 3.12, win_amd64).
- The companion dataset (`RACMAPPO_Dataset`) contains the raw logs behind
  every reported number and figure.

# Reproducing the Paper

Every number in the paper traces to one of the commands below. All runs use
the unified entry points; nothing is hand-edited. Hardware reference: 15
concurrent 3v3 training runs take ~3.5 h each on a desktop CPU + RTX 3060.

## 1. Main table (3v3, Table 2)

Train each algorithm for 1000 episodes, seeds 42–46:

```bash
# BCA (ours) — mission reward only
python train.py --algorithm bca --n_red 3 --n_blue 3 --episodes 1000 \
    --lr 1e-4 --entropy 0.001 --blue_difficulty combat --reward_fn base \
    --eval_interval 50 --eval_episodes 50 --eval_deterministic \
    --save_interval 100 --run_name bca_noshape --seed 42

# Baselines — identical protocol, lr 3e-4, entropy 0.01
python train.py --algorithm {ippo,maddpg,sac,mappo,happo,mat} ... --seed 42
```

Build SWA checkpoints (average of ep600–ep1000) and re-evaluate:

```bash
python swa_build_rev3.py
python reeval_deterministic.py --runs bca_noshape --algo bca --suffix swa \
    --episodes 100 --out results/reeval_bca_noshape_swa.csv
python reeval_deterministic.py --runs ippo,maddpg,sac,mappo,happo,mat \
    --suffix swa --episodes 100 --out results/reeval_baselines_swa.csv
```

## 2. Extended 15-seed study (Section 6.1)

Extend BCA and IPPO to seeds 42–56 (same protocol), then:

```bash
python reeval_deterministic.py --runs bca_base --algo bca --suffix swa \
    --episodes 100 --out results/reeval_rev_bca_swa.csv      # seeds 47–56
python reeval_deterministic.py --runs ippo --suffix swa \
    --episodes 100 --out results/reeval_rev_ippo_swa.csv     # seeds 42–56
```

Statistics (mean/median/IQR, Welch t, Wilcoxon, Mann–Whitney, Levene,
20k-resample bootstrap CI) are computed from the two CSVs; see the analysis
snippet in `results/DONE_REPORT.md`.

## 3. Ablation (Table 3)

```bash
# w/o GRE        : train.py --algorithm bca --no_gre  ... --run_name bca_nogre
# w/o BVD        : train.py --algorithm bca --no_bvd  ... --run_name bca_nobvd_base
# w/o PBC        : train.py --algorithm bca --no_pbc  ... --run_name bca_nopbc
# w/ shaping     : train.py --algorithm bca --reward_fn tactical ... --run_name bca_full
```

then SWA + `reeval_deterministic.py` as above (`--algo bca` for all variants;
`--no_bvd` only removes the competitive critic, the actor interface is unchanged).

## 4. Fairness check (Section 6.2)

```bash
python train.py --algorithm ippo --lr 1e-4 --entropy 0.001 ... --run_name ippo_lr1e4
python reeval_deterministic.py --runs ippo_lr1e4 --algo ippo --suffix swa \
    --episodes 100 --out results/reeval_rev_ippo_lr1e4_swa.csv
```

## 5. Opponent robustness (Section 6.x)

```bash
python reeval_deterministic.py --runs bca_noshape --algo bca --suffix swa \
    --episodes 100 --difficulty easy     --out results/reeval_robust_easy.csv
python reeval_deterministic.py --runs bca_noshape --algo bca --suffix swa \
    --episodes 100 --difficulty maneuver --out results/reeval_robust_maneuver.csv
```

## 6. 5v5 scalability (Table 4)

Identical to Section 1 with `--n_red 5 --n_blue 5`, same hyperparameters.

## Notes

- The `--seed_base 20000` default of `reeval_deterministic.py` makes episode
  initializations identical across algorithms (paired comparison).
- `--difficulty combat` is the evaluation-strength opponent everywhere;
  other levels appear only in the robustness study.

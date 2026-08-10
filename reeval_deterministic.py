#!/usr/bin/env python3
"""
reeval_deterministic.py
=======================
Final deterministic re-evaluation of trained best checkpoints for the paper.

Protocol (symmetric, paired):
  - For every algorithm and every model seed, evaluate the SAME set of
    episode initializations: episode e uses np.random.seed(seed_base + e)
    with env's internal reseeding disabled (common random numbers), and
    deterministic policy action selection (mean of the Gaussian).
  - 100 episodes per (run, model seed) by default.
  - Results are appended incrementally to a CSV so partial progress is kept.

Usage:
  python reeval_deterministic.py --runs random,ippo,maddpg,mappo,happo,mat,sac,sao_mat \
      --episodes 100 --out results/reeval_3v3.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch

from env import CombatEnv
from algorithms import create_policy
from train import judge_result_at_truncation


def find_checkpoints(models_dir, run, suffix='best'):
    """Return list of (seed, path) for <run>_seed<S>_<suffix>.pt, or single-seed fallback."""
    out = []
    for path in sorted(glob.glob(os.path.join(models_dir, f'{run}_seed*_{suffix}.pt'))):
        m = re.search(r'_seed(\d+)_' + re.escape(suffix) + r'\.pt$', path)
        if m:
            out.append((int(m.group(1)), path))
    if not out:
        single = os.path.join(models_dir, f'{run}_{suffix}.pt')
        if os.path.exists(single):
            out.append((-1, single))
    return out


def eval_one(policy, run, n_red, n_blue, episodes, rollout_steps, seed_base):
    """Evaluate one loaded policy deterministically with common random numbers."""
    env = CombatEnv(n_red=n_red, n_blue=n_blue, blue_difficulty='combat',
                    debug_print=False, reward_fn='base')

    # Disable env's internal np.random.seed(None) so episode layouts are
    # reproducible and IDENTICAL across algorithms (paired comparison).
    real_seed = np.random.seed
    np.random.seed = lambda *a, **k: None
    try:
        wins, losses, draws = 0, 0, 0
        kills_red, kills_blue = 0, 0
        returns = []
        for ep in range(episodes):
            real_seed(seed_base + ep)
            torch.manual_seed(seed_base + ep)
            obs = env.reset()
            done, step, ep_ret = False, 0, 0.0
            info = {}
            while not done and step < rollout_steps:
                if not np.isfinite(obs).all():
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                obs = np.clip(obs, -1e6, 1e6)
                with torch.no_grad():
                    try:
                        actions, _ = policy.select_actions(obs, deterministic=True)
                    except TypeError:
                        actions, _ = policy.select_actions(obs)
                obs, rewards, done, info = env.step(actions, None)
                ep_ret += float(np.mean(rewards))
                step += 1
            result = judge_result_at_truncation(info, done)
            if result == 'win':
                wins += 1
            elif result == 'loss':
                losses += 1
            else:
                draws += 1
            kills_red += info.get('kills_red', 0)
            kills_blue += info.get('kills_blue', 0)
            returns.append(ep_ret)
    finally:
        np.random.seed = real_seed

    return {
        'win_rate': wins / episodes * 100.0,
        'avg_return': float(np.mean(returns)),
        'klr': kills_red / max(kills_blue, 1),
        'wins': wins, 'losses': losses, 'draws': draws,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', default='models')
    ap.add_argument('--runs', default='random,ippo,maddpg,mappo,happo,mat,sac,sao_mat')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--n_red', type=int, default=3)
    ap.add_argument('--n_blue', type=int, default=3)
    ap.add_argument('--rollout_steps', type=int, default=800)
    ap.add_argument('--seed_base', type=int, default=20000)
    ap.add_argument('--suffix', default='best',
                    help="checkpoint suffix: 'best' or 'final'")
    ap.add_argument('--algo', default=None,
                    help="override algorithm name for policy construction "
                         "(e.g. ablation run sao_no_order loads into 'sao_mat')")
    ap.add_argument('--out', default='results/reeval_3v3.csv')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    done_keys = set()
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for row in csv.DictReader(f):
                done_keys.add((row['run'], row['model_seed']))

    runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    rows_new = []
    for run in runs:
        algo = args.algo or run
        ckpts = [(0, None)] if run == 'random' else find_checkpoints(args.models_dir, run, args.suffix)
        if not ckpts:
            print(f'[SKIP] {run}: no checkpoints in {args.models_dir}')
            continue
        for seed, path in ckpts:
            if (run, str(seed)) in done_keys:
                print(f'[SKIP] {run} seed{seed}: already in {args.out}')
                continue
            # Build env once to learn obs_dim
            env0 = CombatEnv(n_red=args.n_red, n_blue=args.n_blue,
                             blue_difficulty='combat', debug_print=False, reward_fn='base')
            obs_dim = env0.reset().shape[1]
            del env0
            policy = create_policy(
                algorithm=algo, obs_dim=obs_dim, action_dim=4, n_agents=args.n_red,
                n_red=args.n_red, n_blue=args.n_blue, n_fire_targets=0,
                lr=3e-4,  # unified protocol: all algorithms incl. SAO-MAT use 3e-4
            )
            if path is not None:
                policy.load(path)
            metrics = eval_one(policy, run, args.n_red, args.n_blue,
                               args.episodes, args.rollout_steps, args.seed_base)
            row = {'run': run, 'model_seed': seed, 'episodes': args.episodes,
                   'win_rate': f"{metrics['win_rate']:.1f}",
                   'avg_return': f"{metrics['avg_return']:.2f}",
                   'klr': f"{metrics['klr']:.3f}",
                   'wins': metrics['wins'], 'losses': metrics['losses'],
                   'draws': metrics['draws']}
            rows_new.append(row)
            # incremental append
            write_header = not os.path.exists(args.out)
            with open(args.out, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            print(f"[DONE] {run} seed{seed}: WR={metrics['win_rate']:.1f}% "
                  f"KLR={metrics['klr']:.3f} R={metrics['avg_return']:.1f}")

    # summary across model seeds
    if os.path.exists(args.out):
        import pandas as pd
        df = pd.read_csv(args.out)
        print('\n==== Deterministic re-eval summary (mean +/- std over model seeds) ====')
        for run in runs:
            sub = df[df['run'] == run]
            if len(sub) == 0:
                continue
            wr, klr = sub['win_rate'], sub['klr']
            print(f"  {run:8s} n={len(sub)}  WR={wr.mean():.1f}+/-{wr.std():.1f}%  "
                  f"KLR={klr.mean():.2f}+/-{klr.std():.2f}")


if __name__ == '__main__':
    main()

"""Scripted-ladder evaluation for one RAC-MAPPO checkpoint (paper07, E5).

  python run_ladder_eval.py --ckpt <prefix> --level L3 --episodes 40 \
      --seed0 731000 --out runs/selfplay/e5_L3_a.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.envs.marl_env import MarlEnv
from sim.marl.rac_mappo import RACMAPPO

CFG = 'configs/scenario_2v2_hetero_train.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--level', default='L2')
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--seed0', type=int, default=730000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    probe = MarlEnv(CFG, seed=999999)
    roles = {a: m['spec'].role for a, m in probe.red.items()}
    od = len(probe._obs_all()['red0'])
    probe.close()
    algo = RACMAPPO(roles, od, len(roles))
    algo.load(args.ckpt)

    wins = {'red': 0, 'blue': 0, 'draw': 0}
    durs = []
    for k in range(args.episodes):
        env = MarlEnv(CFG, seed=args.seed0 + k, blue_level=args.level)
        obs = env._obs_all()
        done = False
        while not done:
            gs = env.global_state(obs)
            obs_b = {a: obs[a][None, :] for a in sorted(obs)}
            acts, _ = algo.act(obs_b, gs[None, :], deterministic=True)
            obs, rew, done, info = env.step({a: acts[a][0] for a in acts})
        r = info['result']
        wins[r['winner']] = wins.get(r['winner'], 0) + 1
        durs.append(r['duration_s'])
        env.close()

    n = args.episodes
    out = dict(ckpt=args.ckpt, level=args.level, episodes=n,
               win_rate_red=wins.get('red', 0) / n, wins=wins,
               mean_duration=float(np.mean(durs)))
    print(json.dumps(out, indent=1))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()

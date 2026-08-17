"""Self-play / learned-opponent evaluation (paper07).

Red is flown by one RAC-MAPPO checkpoint, blue by another (or the same,
for mirror self-play). Reports red/blue/draw wins, mean duration, kills.

Examples
--------
  # mirror self-play: doctrine checkpoint vs itself
  python run_selfplay_eval.py --red runs/rac_curr_s0/models/iter_00175 ^
      --blue runs/rac_curr_s0/models/iter_00175 --episodes 100

  # head-to-head: self-play-trained gen1 vs doctrine gen0
  python run_selfplay_eval.py --red runs/sp_gen1_s0/models/iter_00500 ^
      --blue runs/rac_curr_s0/models/iter_00175 --episodes 100
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


def load_algo(ckpt, roles, od, n):
    algo = RACMAPPO(roles, od, n)
    algo.load(ckpt)
    return algo


def run_episode(seed, red_algo, blue_ckpt, record_dir=None):
    env = MarlEnv(CFG, seed=seed, blue_policy=blue_ckpt,
                  run_dir=record_dir, episode_id=0,
                  record=record_dir is not None)
    obs = env._obs_all()
    done = False
    while not done:
        gs = env.global_state(obs)
        obs_b = {a: obs[a][None, :] for a in sorted(obs)}
        acts, _ = red_algo.act(obs_b, gs[None, :], deterministic=True)
        obs, rew, done, info = env.step({a: acts[a][0] for a in acts})
    env.close()
    return info['result']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--red', required=True)
    ap.add_argument('--blue', required=True)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed0', type=int, default=700000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    probe = MarlEnv(CFG, seed=999999)
    roles = {a: m['spec'].role for a, m in probe.red.items()}
    od = len(probe._obs_all()['red0'])
    probe.close()
    red_algo = load_algo(args.red, roles, od, len(roles))

    wins = {'red': 0, 'blue': 0, 'draw': 0}
    durs, causes = [], {}
    kills_r, kills_b = [], []
    for k in range(args.episodes):
        r = run_episode(args.seed0 + k, red_algo, args.blue)
        wins[r['winner']] = wins.get(r['winner'], 0) + 1
        durs.append(r['duration_s'])
        causes[r['cause']] = causes.get(r['cause'], 0) + 1
        kills_r.append(r.get('kills', {}).get('red', 0))
        kills_b.append(r.get('kills', {}).get('blue', 0))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{args.episodes}] wins={wins}", flush=True)

    n = args.episodes
    out = dict(red_ckpt=args.red, blue_ckpt=args.blue, episodes=n,
               seed0=args.seed0,
               win_rate_red=wins.get('red', 0) / n,
               win_rate_blue=wins.get('blue', 0) / n,
               draw_rate=wins.get('draw', 0) / n,
               wins=wins, causes=causes,
               mean_duration=float(np.mean(durs)),
               mean_kills_red=float(np.mean(kills_r)),
               mean_kills_blue=float(np.mean(kills_b)))
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)
        print('saved', args.out)


if __name__ == '__main__':
    main()

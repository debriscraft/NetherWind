"""Multi-seed point-mass fidelity-control trainings (paper04).

Extends the single-seed (seed 2) point-mass runs with seeds 0 and 1,
same protocol as the main pipeline: 500 PPO iterations, n-envs 3,
rollout 64, eval every 10 (5 episodes), opponent curriculum
(--blue-level mixed), point-mass dynamics config.

Wave per seed: RAC-MAPPO, IPPO, RAC-MAPPO w/o role embedding.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs('logs', exist_ok=True)
PY = sys.executable
PM_CFG = 'configs/scenario_2v2_hetero_train_pm.yaml'

ALGOS = [
    ('pm_rac_mappo', 'rac_mappo', []),
    ('pm_ippo', 'ippo', []),
    ('pm_no_roleemb', 'rac_mappo', ['--no-role-emb']),
]


def launch(seed):
    procs = []
    for base, algo, extra in ALGOS:
        name = f'{base}_s{seed}'
        log = open(f'logs/train_{name}.log', 'w')
        p = subprocess.Popen(
            [PY, 'train/run_train.py', '--run-name', name, '--algo', algo,
             '--config', PM_CFG,
             '--n-envs', '3', '--rollout', '64', '--iterations', '500',
             '--eval-every', '10', '--eval-episodes', '5',
             '--blue-level', 'mixed', '--seed', str(seed), *extra],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=0x00000008 | 0x00000200, cwd=ROOT)
        procs.append((name, p))
        print('launched', name, 'pid', p.pid, flush=True)
    return procs


for seed in (0, 1):
    procs = launch(seed)
    for name, p in procs:
        p.wait()
        print('finished', name, 'rc', p.returncode, flush=True)
print('all pm multi-seed trainings done')

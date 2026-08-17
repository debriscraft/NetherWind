"""Launch ablation trainings for RAC-MAPPO (paper04).

Wave 1 (3 runs in parallel): w/o shared role actor, w/o set-attention
critic, w/o track sharing.  Wave 2 (2 runs): w/o team rewards, w/o
curriculum.  Each run: 500 PPO iterations, seed 2, identical to the
full-model training protocol.
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.makedirs('logs', exist_ok=True)
PY = sys.executable

WAVE1 = [
    ('abl_no_role',    'rac2_mappo', ['--blue-level', 'mixed']),
    ('abl_no_attn',    'rac_mappo',  ['--blue-level', 'mixed', '--no-attn-critic']),
    ('abl_no_share',   'rac_mappo',  ['--blue-level', 'mixed', '--no-share-tracks']),
]
WAVE2 = [
    ('abl_no_teamrew', 'rac_mappo',  ['--blue-level', 'mixed', '--no-team-rewards']),
    ('abl_no_curr',    'rac_mappo',  ['--blue-level', 'L2']),
]


def launch(jobs):
    procs = []
    for name, algo, extra in jobs:
        log = open(f'logs/train_{name}.log', 'w')
        p = subprocess.Popen(
            [PY, 'train/run_train.py', '--run-name', name, '--algo', algo,
             '--n-envs', '3', '--rollout', '64', '--iterations', '500',
             '--eval-every', '10', '--eval-episodes', '5',
             '--seed', '2', *extra],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=0x00000008 | 0x00000200, cwd=ROOT)
        procs.append((name, p))
        print('launched', name, 'pid', p.pid, flush=True)
    return procs


def wait(procs):
    for name, p in procs:
        p.wait()
        print('finished', name, 'rc', p.returncode, flush=True)


wave = sys.argv[1] if len(sys.argv) > 1 else 'all'
if wave in ('1', 'all'):
    wait(launch(WAVE1))
if wave in ('2', 'all'):
    wait(launch(WAVE2))
print('ablation wave(s) done')

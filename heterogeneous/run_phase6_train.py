import os, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('logs', exist_ok=True)
jobs = [
    ('rac_mappo', 'rac_mappo', []),                                        # 实验组
    ('b1_mappo_nocoord', 'mappo', ['--no-share-tracks', '--no-team-rewards']),  # 基线1
    ('b3_ippo', 'ippo', []),                                               # 基线3
]
for name, algo, extra in jobs:
    log = open(f'logs/train_{name}.log', 'w')
    subprocess.Popen([sys.executable, 'train/run_train.py',
        '--run-name', name, '--algo', algo,
        '--n-envs', '3', '--rollout', '64', '--iterations', '400',
        '--eval-every', '20', '--eval-episodes', '6',
        '--blue-level', 'L2', '--seed', '0', *extra],
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=0x00000008 | 0x00000200, cwd='.')
    print('launched', name)

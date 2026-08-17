import os, subprocess, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('logs', exist_ok=True)
CKPT = 'runs/rac_curr_s2/models/iter_00200'
jobs = [
    ('scale3v2_ft',     'configs/scenario_3v2_hetero.yaml', ['--load', CKPT]),
    ('scale3v2_scratch','configs/scenario_3v2_hetero.yaml', []),
    ('scale6v6_ft',     'configs/scenario_6v6_hetero.yaml', ['--load', CKPT]),
    ('scale6v6_scratch','configs/scenario_6v6_hetero.yaml', []),
]
for name, cfg, extra in jobs:
    log = open(f'logs/train_{name}.log', 'w')
    p = subprocess.Popen([sys.executable, 'train/run_train.py',
        '--run-name', name, '--algo', 'rac_mappo', '--config', cfg,
        '--n-envs', '4', '--rollout', '64', '--iterations', '500',
        '--eval-every', '20', '--eval-episodes', '6',
        '--blue-level', 'mixed', '--seed', '0', *extra],
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=0x00000008 | 0x00000200, cwd='.')
    print('launched', name, 'pid', p.pid)
    time.sleep(20)   # stagger startup to avoid spawn storms

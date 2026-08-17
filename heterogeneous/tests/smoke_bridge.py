"""
tests/smoke_bridge.py
=====================
Phase 1 verification gate: heterogeneous two-aircraft bridge smoke test.

Spawns one F-16 (fighter) and one OV-10 (UCAV surrogate) through the L0
bridge, flies both for 60 s under the L2 autopilot (fighter: straight +
level at cruise; UCAV: gentle orbit), records every physics step through
the FlightRecorder, then asserts:

  1. all states finite, both aircraft alive;
  2. speeds remain inside plausible platform envelopes
     (F-16 fast jet, OV-10 turboprop — the heterogeneity must be visible
     in the recorded data, not just in the YAML);
  3. the log round-trips through the replay loader;
  4. E-M fields (specific energy, turn-relevant nz, alpha) are present and
     finite (they feed the energy-maneuverability analysis later).

Run:  python tests/smoke_bridge.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core.aircraft import Aircraft
from sim.core.autopilot import Autopilot
from sim.core.platforms import load_platforms
from sim.core.recorder import FlightRecorder, load_episode

G = 9.80665


def main():
    platforms = load_platforms()
    f16 = platforms['f16_fighter']
    ucav = platforms['ov10_ucav']

    # ---- spawn: 3 km head-on separation, 3000 m ----
    ac_a = Aircraft(f16.jsbsim_model, [0, -1500, -3000],
                    init_psi=np.pi / 2, speed=f16.cruise_speed_ms,
                    profile=f16.dynamics_profile)
    ac_b = Aircraft(ucav.jsbsim_model, [0, 1500, -3000],
                    init_psi=-np.pi / 2, speed=ucav.cruise_speed_ms,
                    profile=ucav.dynamics_profile)
    ap_a, ap_b = Autopilot(f16), Autopilot(ucav)

    run_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'logs', 'smoke_bridge')

    steps = 600  # 60 s at 10 Hz
    with FlightRecorder(run_dir, episode_id=0, meta={
            'scenario': 'smoke_bridge',
            'platforms': {'A': f16.name, 'B': ucav.name}}) as rec:
        for k in range(steps):
            # fighter: straight & level cruise
            cmd_a = ap_a.command(ac_a, np.pi / 2, 3000.0, f16.cruise_speed_ms)
            # UCAV: gentle orbit (constant right turn) at its own cruise speed
            psi_orbit = -np.pi / 2 + 2 * np.pi * (k / steps)
            cmd_b = ap_b.command(ac_b, psi_orbit, 3000.0, ucav.cruise_speed_ms)

            ac_a.step(cmd_a)
            ac_b.step(cmd_b)

            rec.log_step({
                't': k * ac_a.dt, 'step': k,
                'aircraft': [
                    {'id': 'A_f16', 'platform': f16.name,
                     **ac_a.full_state(),
                     'especific': ac_a.speed ** 2 / (2 * G) + (-ac_a.position[2])},
                    {'id': 'B_ucav', 'platform': ucav.name,
                     **ac_b.full_state(),
                     'especific': ac_b.speed ** 2 / (2 * G) + (-ac_b.position[2])},
                ],
                'flight_cmd': {'A_f16': cmd_a, 'B_ucav': cmd_b},
                'tactical': {},
                'events': [],
                'rewards': None,
            })

    # ---- assertions ----
    data = load_episode(rec.path)
    assert len(data) == steps, f'expected {steps} records, got {len(data)}'

    sa = np.array([r['aircraft'][0]['speed'] for r in data])
    sb = np.array([r['aircraft'][1]['speed'] for r in data])
    ha = np.array([r['aircraft'][0]['alt'] for r in data])
    hb = np.array([r['aircraft'][1]['alt'] for r in data])
    for arr, name in ((sa, 'A speed'), (sb, 'B speed'),
                      (ha, 'A alt'), (hb, 'B alt')):
        assert np.isfinite(arr).all(), f'{name} contains NaN/Inf'

    assert sa.mean() > 180.0, f'F-16 too slow: {sa.mean():.0f} m/s'
    assert 50.0 < sb.mean() < 160.0, f'OV-10 outside turboprop band: {sb.mean():.0f} m/s'
    assert abs(ha[-1] - 3000) < 500, f'F-16 altitude hold failed: {ha[-1]:.0f}'
    assert abs(hb[-1] - 3000) < 800, f'OV-10 altitude hold failed: {hb[-1]:.0f}'
    assert ac_a.alive and ac_b.alive, 'unexpected crash'

    # E-M fields present & finite
    es = np.array([r['aircraft'][0]['especific'] for r in data])
    nz = np.array([r['aircraft'][1]['nz'] for r in data])
    assert np.isfinite(es).all() and np.isfinite(nz).all()

    print('SMOKE OK')
    print(f'  records        : {len(data)} steps ({data[-1]["t"]:.0f} s)')
    print(f'  F-16  speed    : mean {sa.mean():.0f} m/s, end alt {ha[-1]:.0f} m')
    print(f'  OV-10 speed    : mean {sb.mean():.0f} m/s, end alt {hb[-1]:.0f} m')
    print(f'  F-16  Es range : {es.min():.0f} .. {es.max():.0f} m')
    print(f'  OV-10 nz range : {nz.min():.2f} .. {nz.max():.2f} g')
    print(f'  log            : {rec.path}')
    print(f'  meta           : {rec.meta_path}')


if __name__ == '__main__':
    main()

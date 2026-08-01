#!/usr/bin/env python3

# Pedestrian UWB agent.
#
# Each walker UWB:
#   - listen() 콜백에서 infrastructure(anchor / traffic_light) 측정값만 수락.
#     차량·보행자 신호는 payload type 체크로 즉시 버림.
#   - 가장 가까운 4개 인프라 노드로 trilateration → 자기 위치 추정.
#   - 추정 위치를 send_uwb_payload로 송신: {type, id, x, y, z}
#
# UWB 센서는 sensor_attach.py 가 이미 부착해 뒀다고 가정.

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla

_INFRA_TYPES = {'anchor', 'traffic_light'}
_MAX_ANCHORS = 6


def _multilaterate(points):
    if len(points) < 3:
        return None
    import numpy as np
    x1, y1, z1, d1 = points[0]
    A, b = [], []
    for xi, yi, zi, di in points[1:]:
        A.append([2*(xi - x1), 2*(yi - y1), 2*(zi - z1)])
        b.append(d1**2 - di**2
                 + xi**2 - x1**2
                 + yi**2 - y1**2
                 + zi**2 - z1**2)
    pos, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return pos


def parse_args():
    p = argparse.ArgumentParser(description='Pedestrian UWB agent')
    p.add_argument('--host',      default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    return p.parse_args()


# ── Callback factory ──────────────────────────────────────────────────────────

def _make_callback(uwb, walker):
    def on_measure(measurement):
        # 인프라 노드 측정값만 추출 (payload type 확인)
        infra = [
            det for det in measurement
            if det.payload and det.payload.get('type') in _INFRA_TYPES
        ]
        if not infra:
            return

        # 가장 가까운 4개
        nearest = sorted(infra, key=lambda d: d.distance)[:_MAX_ANCHORS]

        points = []
        for det in nearest:
            p = det.payload
            try:
                points.append((
                    float(p['x']), float(p['y']), float(p['z']),
                    det.distance,
                ))
            except (KeyError, TypeError, ValueError):
                continue

        pos = _multilaterate(points)
        if pos is None:
            return

        uwb.send_uwb_payload({
            'type': 'pedestrian',
            'id':   walker.id,
            'x':    round(float(pos[0]), 3),
            'y':    round(float(pos[1]), 3),
            'z':    round(float(pos[2]), 3),
        })

    return on_measure


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world  = client.get_world()

    registered_uwbs = []

    def cleanup(sig=None, frame=None):
        print(f'\n[pedestrian] Stopping {len(registered_uwbs)} listeners...')
        for s in reversed(registered_uwbs):
            try:
                if s.is_alive:
                    s.stop()
            except Exception:
                pass
        print('[pedestrian] Done.')
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # sensor_attach.py 가 보행자에 센서 부착을 완료할 때까지 대기.
    # UWB 개수가 2초 연속 동일하면 부착 완료로 판단.
    # generate_traffic.py 가 실행 중이 아니어도(보행자 0대) 넘어갈 수 있도록,
    # 0대인 상태도 안정 상태로 인정한다.
    print('[pedestrian] 보행자 및 센서 대기 중 (보행자 0대여도 진행)...')
    prev_uwb_cnt = -1
    stable_ticks = 0
    while stable_ticks < 2:
        walker_ids = {w.id for w in world.get_actors().filter('walker.pedestrian.*')}
        uwb_cnt = sum(
            1 for s in world.get_actors().filter('sensor.other.uwb')
            if s.parent is not None and s.parent.type_id.startswith('walker.')
        )
        if uwb_cnt == prev_uwb_cnt:
            stable_ticks += 1
        else:
            stable_ticks = 0
        prev_uwb_cnt = uwb_cnt
        print(f'\r  보행자={len(walker_ids)}  UWB={uwb_cnt} ', end='', flush=True)
        time.sleep(1.0)
    print()

    # sensor_attach.py 가 walker에 붙여 둔 UWB 센서 수집
    walker_to_uwb = {
        s.parent.id: s
        for s in world.get_actors().filter('sensor.other.uwb')
        if s.parent and s.parent.type_id.startswith('walker.')
    }

    walkers = list(world.get_actors().filter('walker.pedestrian.*'))
    print(f'[pedestrian] {len(walkers)} walkers  |  {len(walker_to_uwb)} UWBs from sensor_attach\n')

    ok = 0
    for walker in walkers:
        uwb = walker_to_uwb.get(walker.id)
        if uwb is None:
            print(f'  walker {walker.id:>6}: UWB not found — run sensor_attach.py first')
            continue

        uwb.listen(_make_callback(uwb, walker))
        registered_uwbs.append(uwb)
        ok += 1
        print(f'  walker {walker.id:>6}  UWB={uwb.id:>8}  registered')

    print(f'\n  → {ok} walkers active')
    print('[pedestrian] Running... Ctrl+C to stop')

    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()

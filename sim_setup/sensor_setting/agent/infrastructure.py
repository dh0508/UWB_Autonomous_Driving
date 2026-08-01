#!/usr/bin/env python3

# Infrastructure UWB agent — send-only, no receive processing.
#
# Two roles:
#   Roadside anchors  – spawned world-fixed by sensor_attach.py.
#                       Payload set once at startup (position never changes).
#   Traffic lights    – UWB attached here; payload refreshed every loop tick
#                       so signal state stays current.
#
# send_uwb_payload() is called directly — no listen() callback,
# so received measurements are never processed.

import argparse
import signal
import sys
import time

import carla

_STATE = {
    carla.TrafficLightState.Red:     'Red',
    carla.TrafficLightState.Yellow:  'Yellow',
    carla.TrafficLightState.Green:   'Green',
    carla.TrafficLightState.Off:     'Off',
    carla.TrafficLightState.Unknown: 'Unknown',
}


def parse_args():
    p = argparse.ArgumentParser(description='Infrastructure UWB agent (send-only)')
    p.add_argument('--host',      default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    p.add_argument('--uwb-range', default=150.0, type=float)
    return p.parse_args()


def _make_uwb_bp(bp_lib, max_range: float):
    bp = bp_lib.find('sensor.other.uwb')
    bp.set_attribute('max_range',           str(max_range))
    bp.set_attribute('noise_los_stddev',    '0.03')
    bp.set_attribute('noise_nlos_bias_min', '0.1')
    bp.set_attribute('noise_nlos_bias_max', '1.0')
    bp.set_attribute('noise_nlos_stddev',   '0.2')
    return bp


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    owned_sensors = []

    def cleanup(sig=None, frame=None):
        print(f'\n[infrastructure] Destroying {len(owned_sensors)} sensors...')
        for s in reversed(owned_sensors):
            try:
                if s.is_alive:
                    s.destroy()
            except Exception:
                pass
        print('[infrastructure] Done.')
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    uwb_bp = _make_uwb_bp(bp_lib, args.uwb_range)

    # sensor_attach.py 가 도로앵커를 스폰할 때까지 대기
    print('[infrastructure] 도로앵커 대기 중...')
    while True:
        anchors_check = [
            a for a in world.get_actors().filter('sensor.other.uwb')
            if a.parent is None
        ]
        if anchors_check:
            break
        time.sleep(1.0)

    # ── Roadside anchors: collect (sensor, fixed_payload) tuples ─────────
    print('[infrastructure] Registering roadside anchors...')
    anchors = [
        a for a in world.get_actors().filter('sensor.other.uwb')
        if a.parent is None
    ]
    anchor_sensors = []   # [(uwb_sensor, fixed_payload), ...]
    for s in anchors:
        loc = s.get_transform().location
        payload = {
            'type': 'anchor',
            'id':   s.id,
            'x':    round(loc.x, 3),
            'y':    round(loc.y, 3),
            'z':    round(loc.z, 3),
        }
        anchor_sensors.append((s, payload))
        print(f'  anchor  UWB={s.id:>8}'
              f'  pos=({loc.x:8.2f}, {loc.y:8.2f}, {loc.z:6.2f})')
    print(f'  → {len(anchor_sensors)} anchors\n')

    # ── Traffic lights: attach UWB, collect (sensor, tl, loc) tuples ──────
    print('[infrastructure] Attaching UWB to traffic lights...')
    tl_sensors = []   # [(uwb_sensor, tl_actor, fixed_loc), ...]
    for tl in world.get_actors().filter('traffic.traffic_light'):
        uwb = world.try_spawn_actor(
            uwb_bp,
            carla.Transform(carla.Location(z=3.0)),
            attach_to=tl,
        )
        if uwb:
            owned_sensors.append(uwb)
            loc = tl.get_location()
            tl_sensors.append((uwb, tl, loc))
            print(f'  tl      UWB={uwb.id:>8}  tl_id={tl.id:>6}'
                  f'  pos=({loc.x:8.2f}, {loc.y:8.2f}, {loc.z:6.2f})')
        else:
            print(f'  tl {tl.id}: UWB spawn FAILED')
    print(f'  → {len(tl_sensors)} traffic lights\n')

    print('[infrastructure] Running (send-only)... Ctrl+C to stop')

    # ── Main loop: refresh anchor + TL payloads every simulated tick ─────
    # world.wait_for_tick()으로 실제(wall-clock) 주기가 아니라 서버의 tick 경계에 맞춰
    # 갱신한다. 이전에는 time.sleep(TICK)으로 실시간 0.1초마다만 신호를 갱신했는데,
    # rule_based_test.py 등 다른 클라이언트가 synchronous 모드를 페이싱 없이(배속으로)
    # 돌리면 시뮬레이션이 실시간보다 훨씬 빨리 진행돼 여기서 보내는 signal 값이 실제
    # 상태보다 뒤처졌다 — 신호등이 화면상 초록으로 바뀐 뒤에도 hero가 계속 정지해 있던
    # 원인. wait_for_tick은 sync/async 모드 어느 쪽이든 다음 tick의 스냅샷이 준비될
    # 때까지 블록하므로, 배속으로 돌든 실시간으로 돌든 매 tick마다 정확히 그 순간의
    # signal이 방송된다.
    while True:
        world.wait_for_tick()
        for uwb, payload in anchor_sensors:
            uwb.send_uwb_payload(payload)
        for uwb, tl, loc in tl_sensors:
            uwb.send_uwb_payload({
                'type':   'traffic_light',
                'tl_id':  tl.id,
                'x':      round(loc.x, 3),
                'y':      round(loc.y, 3),
                'z':      round(loc.z, 3),
                'signal': _STATE.get(tl.get_state(), 'Unknown'),
            })


if __name__ == '__main__':
    main()

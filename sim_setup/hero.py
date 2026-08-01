#!/usr/bin/env python3

# Hero vehicle agent — 센서 부착 + 대시보드 표시 전용.
#
# 추정 로직은 agent/vehicle.py 의 _setup_vehicle 을 그대로 사용(send=False).
# hero.py 는 센서를 부착하고, 추정 결과를 대시보드로 렌더링하는 역할만 한다.
#
# 전체 스택 실행 순서 (각 터미널에서 별도 실행):
#   1. generate_traffic.py
#   2. sensor_attach.py
#   3. agent/infrastructure.py
#   4. agent/vehicle.py
#   5. agent/pedestrian.py
#   6. manual_control.py          ← hero 차량 생성
#   7. hero.py  (이 스크립트)     ← hero 에 센서 부착 + 대시보드

import argparse
import math
import os
import signal
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sensor_setting'))

import carla
from agent.vehicle import _setup_vehicle

_DISPLAY_HZ = 2
W = 78

_TARGET_SAMPLES = 500   # 오차 샘플을 이 개수만큼 모으면 통계 확정 후 저장 종료


def parse_args():
    p = argparse.ArgumentParser(description='Hero vehicle agent')
    p.add_argument('--host',       default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    p.add_argument('--uwb-range',  default=150.0, type=float)
    return p.parse_args()


def _make_uwb_bp(bp_lib, max_range):
    bp = bp_lib.find('sensor.other.uwb')
    bp.set_attribute('max_range',           str(max_range))
    bp.set_attribute('noise_los_stddev',    '0.05')
    bp.set_attribute('noise_nlos_bias_min', '0.2')
    bp.set_attribute('noise_nlos_bias_max', '2.0')
    bp.set_attribute('noise_nlos_stddev',   '0.3')
    return bp


# ── Dashboard helpers ─────────────────────────────────────────────────────────

def _sec(title):
    pad = W - 4 - len(title)
    return f'  ── {title} ' + '─' * max(pad, 0)


def _col_hdr():
    return (f'  {"":12} {"PUB (수신/추정)":>15}  '
            f'│ {"ACTUAL (실제)":>15}  │     오차')


def _row(label, pub, actual, unit=''):
    if pub is None:
        pub_s = f'{"—":>10}'
        err_s = '        —'
    else:
        pub_s = f'{pub:10.3f}'
        err_s = f'{pub - actual:+9.3f}'
    return (f'  {label:<12} {pub_s} {unit:<5}'
            f'│ {actual:10.3f} {unit:<5}'
            f'│ {err_s} {unit}')


# ── Render ────────────────────────────────────────────────────────────────────

def _acc_err(errors, label, pub, actual):
    if pub is not None:
        errors[label].append(abs(pub - actual))


def _render(hero_id, state_ref, lock, actual, errors, record, n_recorded):
    with lock:
        imu   = dict(state_ref['imu'])
        epos  = list(state_ref['est_pos'])
        wheel = dict(state_ref['wheel'])

    # 오차 누적 (절대값) — 저장 구간에서만
    if record:
        _acc_err(errors, 'x',        epos[0],        actual['x'])
        _acc_err(errors, 'y',        epos[1],        actual['y'])
        _acc_err(errors, 'z',        epos[2],        actual['z'])
        _acc_err(errors, 'roll',     imu['roll'],     actual['roll'])
        _acc_err(errors, 'pitch',    imu['pitch'],    actual['pitch'])
        _acc_err(errors, 'yaw',      imu['yaw'],      actual['yaw'])
        _acc_err(errors, 'yaw_rate', imu['yaw_rate'], actual['yaw_rate'])
        _acc_err(errors, 'vel_x',    wheel['vel_x'],  actual['vel_x'])
        _acc_err(errors, 'vel_y',    wheel['vel_y'],  actual['vel_y'])
        _acc_err(errors, 'speed',    wheel['speed'],  actual['speed'])
        _acc_err(errors, 'acc_x',    wheel['acc_x'],  actual['acc_x'])
        _acc_err(errors, 'acc_y',    wheel['acc_y'],  actual['acc_y'])
        _acc_err(errors, 'acc_z',    wheel['acc_z'],  actual['acc_z'])
        _acc_err(errors, 'a_mag',    wheel['a_mag'],  actual['a_mag'])

    lines = []

    # ── Hero ──────────────────────────────────────────────────────────────
    lines.append('═' * W)
    done = n_recorded >= _TARGET_SAMPLES
    tag  = '완료' if done else '수집 중'
    lines.append(f'  HERO  id={hero_id}    오차 샘플: {n_recorded}/{_TARGET_SAMPLES}  [{tag}]')
    lines.append(_col_hdr())
    lines.append(_sec('위치  trilateration (m)'))
    lines.append(_row('x',        epos[0],         actual['x'],        'm'))
    lines.append(_row('y',        epos[1],         actual['y'],        'm'))
    lines.append(_row('z',        epos[2],         actual['z'],        'm'))
    lines.append(_sec('자세  IMU (deg / rad/s)'))
    lines.append(_row('roll',     imu['roll'],      actual['roll'],     '°'))
    lines.append(_row('pitch',    imu['pitch'],     actual['pitch'],    '°'))
    lines.append(_row('yaw',      imu['yaw'],       actual['yaw'],      '°'))
    lines.append(_row('yaw_rate', imu['yaw_rate'],  actual['yaw_rate'], 'r/s'))
    lines.append(_sec('속도  휠 엔코더 (m/s)'))
    lines.append(_row('vel_x',    wheel['vel_x'],   actual['vel_x'],    'm/s'))
    lines.append(_row('vel_y',    wheel['vel_y'],   actual['vel_y'],    'm/s'))
    lines.append(_row('speed',    wheel['speed'],   actual['speed'],    'm/s'))
    lines.append(_sec('가속도  휠 엔코더 (m/s²)'))
    lines.append(_row('acc_x',    wheel['acc_x'],   actual['acc_x'],    'm/s²'))
    lines.append(_row('acc_y',    wheel['acc_y'],   actual['acc_y'],    'm/s²'))
    lines.append(_row('acc_z',    wheel['acc_z'],   actual['acc_z'],    'm/s²'))
    lines.append(_row('a_mag',    wheel['a_mag'],   actual['a_mag'],    'm/s²'))

    lines.append('═' * W)

    # 500 샘플을 다 모으면 확정된 통계를 대시보드 하단에 계속 띄워 둔다.
    if done:
        lines.extend(_err_summary_lines(errors))

    sys.stdout.write('\033[H\033[2J')
    sys.stdout.write('\n'.join(lines) + '\n')
    sys.stdout.flush()


def _err_summary_lines(errors):
    if not errors:
        return ['  누적된 오차 데이터 없음.']
    out = ['═' * W, '  오차 통계  (절대값, n=샘플 수)',
           f'  {"":12} {"평균":>10}  {"최소":>10}  {"최대":>10}  {"n":>6}']
    for label, vals in errors.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        out.append(f'  {label:<12} {mean:10.3f}  {min(vals):10.3f}  '
                   f'{max(vals):10.3f}  {len(vals):6d}')
    out.append('═' * W)
    return out


def _print_err_summary(errors):
    for line in _err_summary_lines(errors):
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world     = client.get_world()
    bp_lib    = world.get_blueprint_library()
    world_map = world.get_map()   # MAP 제약(도로망 매칭)에 사용

    owned_sensors = []
    errors = defaultdict(list)

    def cleanup(sig=None, frame=None):
        print('\n[hero] 정리 중...')
        _print_err_summary(errors)
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            world.apply_settings(s)
        except Exception:
            pass
        for s in reversed(owned_sensors):
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass
        print('[hero] Done.')
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── hero 차량 대기 ────────────────────────────────────────────────────
    print('[hero] manual_control.py 의 hero 차량 대기 중...')
    hero = None
    while hero is None:
        for a in world.get_actors().filter('vehicle.*'):
            if a.attributes.get('role_name') == 'hero':
                hero = a
                break
        if hero is None:
            time.sleep(0.5)
    print(f'[hero] hero 발견  id={hero.id}  type={hero.type_id}')

    # ── IMU + UWB 부착 ────────────────────────────────────────────────────
    mount = carla.Transform(carla.Location(z=1.5))
    imu   = world.spawn_actor(bp_lib.find('sensor.other.imu'), mount, attach_to=hero)
    uwb   = world.spawn_actor(_make_uwb_bp(bp_lib, args.uwb_range), mount, attach_to=hero)
    owned_sensors += [imu, uwb]
    print(f'[hero] IMU={imu.id}  UWB={uwb.id}  range={args.uwb_range} m')

    # ── 동기 모드 활성화 (hero.py 가 ticker master) ───────────────────────
    _s = world.get_settings()
    _s.synchronous_mode    = True
    _s.fixed_delta_seconds = 0.05
    world.apply_settings(_s)
    print('[hero] 동기 모드 활성화  fixed_delta=0.05 s (20 Hz)')

    # ── 추정 로직 위임 (vehicle.py 와 동일, 송신 없음) ───────────────────
    _, state_ref, lock = _setup_vehicle(hero, uwb, imu, world_map, send=False)

    # ── 대시보드 루프 ─────────────────────────────────────────────────────
    print(f'[hero] 실행 중... 오차 샘플 {_TARGET_SAMPLES}개 수집 (Ctrl+C 로 종료)\n')
    _last_render = 0.0
    _n_recorded  = 0

    while True:
        world.tick()

        if not hero.is_alive:
            print('[hero] hero 차량이 사라짐.')
            cleanup()

        t = time.time()
        if t - _last_render >= 1.0 / _DISPLAY_HZ:
            loc = hero.get_location()
            rot = hero.get_transform().rotation
            vel = hero.get_velocity()
            acc = hero.get_acceleration()
            ang = hero.get_angular_velocity()
            actual = {
                'x':        loc.x,
                'y':        loc.y,
                'z':        loc.z,
                'roll':     rot.roll,
                'pitch':    rot.pitch,
                'yaw':      rot.yaw,
                'yaw_rate': math.radians(ang.z),
                'vel_x':    vel.x,
                'vel_y':    vel.y,
                'speed':    math.sqrt(vel.x**2 + vel.y**2),
                'acc_x':    acc.x,
                'acc_y':    acc.y,
                'acc_z':    acc.z,
                'a_mag':    math.sqrt(acc.x**2 + acc.y**2),
            }
            record = _n_recorded < _TARGET_SAMPLES
            _render(hero.id, state_ref, lock, actual, errors, record, _n_recorded)
            if record:
                _n_recorded += 1
            _last_render = t


if __name__ == '__main__':
    main()

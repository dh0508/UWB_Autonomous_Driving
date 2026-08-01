#!/usr/bin/env python3

# Attach IMU + UWB to every vehicle, UWB to every walker,
# and place roadside UWB anchors every 30 m alternating left/right.
# generate_traffic.py is optional — if it isn't running (0 vehicles/walkers),
# this still installs the roadside anchors so infrastructure-only setups work.
# Sensor callbacks / communication are handled in separate scripts.

import argparse
import signal
import sys
import time

import carla

ROADSIDE_INTERVAL = 30.0   # m between roadside anchors along the road
ROADSIDE_OFFSET   = 5.0    # m from road-center waypoint to anchor (curb-side)
ROADSIDE_HEIGHT   = 4.0    # m above road surface (simulates pole mounting)


def parse_args():
    p = argparse.ArgumentParser(
        description='Attach IMU/UWB to traffic from generate_traffic.py')
    p.add_argument('--host',      default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    p.add_argument('--uwb-range', default=150.0, type=float,
                   help='UWB max_range in metres (default: 150)')
    return p.parse_args()


def _make_uwb_bp(bp_lib, max_range: float):
    bp = bp_lib.find('sensor.other.uwb')
    bp.set_attribute('max_range',           str(max_range))
    bp.set_attribute('noise_los_stddev',    '0.05')
    bp.set_attribute('noise_nlos_bias_min', '0.2')
    bp.set_attribute('noise_nlos_bias_max', '2.0')
    bp.set_attribute('noise_nlos_stddev',   '0.3')
    return bp


def _make_imu_bp(bp_lib):
    return bp_lib.find('sensor.other.imu')


# ── Vehicle sensors ───────────────────────────────────────────────────────────

def attach_vehicle_sensors(world, bp_lib, vehicles, uwb_range: float):
    """Attach one IMU and one UWB to every vehicle. Returns spawned sensors."""
    sensors = []
    imu_bp  = _make_imu_bp(bp_lib)
    uwb_bp  = _make_uwb_bp(bp_lib, uwb_range)
    mount   = carla.Transform(carla.Location(z=1.5))

    for v in vehicles:
        imu = world.try_spawn_actor(imu_bp, mount, attach_to=v)
        if imu:
            sensors.append(imu)

        uwb = world.try_spawn_actor(uwb_bp, mount, attach_to=v)
        if uwb:
            sensors.append(uwb)

        print(f'  vehicle {v.id:>6}  IMU={imu.id if imu else "FAIL":>8}'
              f'  UWB={uwb.id if uwb else "FAIL":>8}')

    return sensors


# ── Walker sensors ────────────────────────────────────────────────────────────

def attach_walker_sensors(world, bp_lib, walkers, uwb_range: float):
    """Attach one UWB to every pedestrian. Returns spawned sensors."""
    sensors = []
    uwb_bp  = _make_uwb_bp(bp_lib, uwb_range)
    mount   = carla.Transform(carla.Location(z=1.0))

    for w in walkers:
        uwb = world.try_spawn_actor(uwb_bp, mount, attach_to=w)
        if uwb:
            sensors.append(uwb)
        print(f'  walker  {w.id:>6}  UWB={uwb.id if uwb else "FAIL":>8}')

    return sensors


# ── Roadside UWB anchors ──────────────────────────────────────────────────────

def spawn_roadside_uwb(world, bp_lib, uwb_range: float):
    """
    Place static UWB anchors every ROADSIDE_INTERVAL metres along the road
    network, alternating right-side / left-side.

    Deduplication key: (road_id, s-bucket) so multi-lane roads only get
    one anchor per 30 m slice, not one per lane.
    """
    world_map = world.get_map()
    uwb_bp    = _make_uwb_bp(bp_lib, uwb_range)

    raw_wps = world_map.generate_waypoints(ROADSIDE_INTERVAL)

    # One representative waypoint per 30 m road slice, skip junction tiles
    seen      = set()
    unique_wps = []
    for wp in raw_wps:
        if wp.is_junction:
            continue
        bucket = (wp.road_id, int(wp.s / ROADSIDE_INTERVAL))
        if bucket not in seen:
            seen.add(bucket)
            unique_wps.append(wp)

    sensors = []
    for idx, wp in enumerate(unique_wps):
        rv  = wp.transform.get_right_vector()
        loc = wp.transform.location

        # Even index → right curb (+offset), odd index → left curb (−offset)
        sign = 1.0 if idx % 2 == 0 else -1.0
        side = 'R' if sign > 0 else 'L'

        anchor_loc = carla.Location(
            x=loc.x + rv.x * sign * ROADSIDE_OFFSET,
            y=loc.y + rv.y * sign * ROADSIDE_OFFSET,
            z=loc.z + ROADSIDE_HEIGHT,
        )
        anchor_tf = carla.Transform(anchor_loc, wp.transform.rotation)

        # Spawned without attach_to → fixed world position
        sensor = world.try_spawn_actor(uwb_bp, anchor_tf)
        if sensor:
            sensors.append(sensor)
            print(f'  roadside[{idx:>3}] {side}  UWB={sensor.id:>8}'
                  f'  ({anchor_loc.x:7.1f}, {anchor_loc.y:7.1f}, {anchor_loc.z:5.1f})')
        else:
            print(f'  roadside[{idx:>3}] {side}  FAIL'
                  f'  ({anchor_loc.x:7.1f}, {anchor_loc.y:7.1f}, {anchor_loc.z:5.1f})')

    return sensors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    # 이전에 hero.py / rule_based_test.py 등이 synchronous_mode=True 로 서버를 바꾼 뒤
    # 강제 종료(크래시·kill)되면 async 복구가 안 돼 서버가 동기모드로 남는다. 그 상태에서는
    # 아무도 world.tick()을 하지 않아 시뮬레이션이 멈추듯 프레임이 계속 드랍된다. setup 은
    # 프레임 진행을 관리하지 않으므로, 시작 시 무조건 asynchronous 모드로 되돌려 놓는다.
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode  = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print('[sensor_attach] 서버가 synchronous 모드로 남아 있어 asynchronous 로 복구함.')

    all_sensors = []

    def cleanup(sig=None, frame=None):
        print(f'\n[sensor_attach] Destroying {len(all_sensors)} sensors...')
        for s in reversed(all_sensors):
            try:
                if s.is_alive:
                    s.destroy()
            except Exception:
                pass
        print('[sensor_attach] Done.')
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # generate_traffic.py 가 차량·보행자를 스폰했다면 그 수가 3초 연속 동일할 때까지 대기.
    # generate_traffic.py 가 실행 중이 아니어도(차량·보행자 0대) 로드사이드 앵커 등
    # 인프라만 설치하고 넘어갈 수 있도록, 0대인 상태도 안정 상태로 인정한다.
    print('[sensor_attach] 차량/보행자 대기 중 (generate_traffic.py 미실행 시 0대로 진행)...')
    prev_cnt    = -1
    stable_ticks = 0
    while stable_ticks < 3:
        vehicles = list(world.get_actors().filter('vehicle.*'))
        walkers  = list(world.get_actors().filter('walker.pedestrian.*'))
        cnt = len(vehicles) + len(walkers)
        if cnt == prev_cnt:
            stable_ticks += 1
        else:
            stable_ticks = 0
        prev_cnt = cnt
        print(f'\r  차량={len(vehicles)}  보행자={len(walkers)} ', end='', flush=True)
        time.sleep(1.0)
    print()

    print(f'[sensor_attach] {len(vehicles)} vehicles  {len(walkers)} walkers\n')

    print('[sensor_attach] Vehicles → IMU + UWB')
    all_sensors += attach_vehicle_sensors(world, bp_lib, vehicles, args.uwb_range)

    print(f'\n[sensor_attach] Walkers → UWB')
    all_sensors += attach_walker_sensors(world, bp_lib, walkers, args.uwb_range)

    print(f'\n[sensor_attach] Roadside anchors → UWB every {ROADSIDE_INTERVAL:.0f} m (alt. L/R)')
    all_sensors += spawn_roadside_uwb(world, bp_lib, args.uwb_range)

    print(f'\n[sensor_attach] Total sensors: {len(all_sensors)}  |  Ctrl+C to destroy and exit\n')

    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

# CARLA HD-map downloader.
#
# 실행하면 두 파일을 data/map/ 에 저장 (data/src/paths.py의 MAP_JSON과 동일 경로):
#   <map_name>.xodr     — OpenDRIVE 원본 (CARLA 제공)
#   <map_name>.json     — 학습용 레이어 (waypoints, lanes, junctions, traffic_lights)
#
# 사용:
#   python sim_setup/map_downloader.py [--host 127.0.0.1] [-p 2000] [--interval 2.0]

import argparse
import json
import os
import sys

import carla

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_DIR      = os.path.join(_PROJECT_ROOT, 'data', 'map')

_LANE_TYPE_STR = {
    carla.LaneType.Driving:        'driving',
    carla.LaneType.Shoulder:       'shoulder',
    carla.LaneType.Sidewalk:       'sidewalk',
    carla.LaneType.Biking:         'biking',
    carla.LaneType.Parking:        'parking',
    carla.LaneType.Stop:           'stop',
    carla.LaneType.Restricted:     'restricted',
    carla.LaneType.Border:         'border',
    carla.LaneType.Any:            'any',
}

_TL_STATE_STR = {
    carla.TrafficLightState.Red:     'Red',
    carla.TrafficLightState.Yellow:  'Yellow',
    carla.TrafficLightState.Green:   'Green',
    carla.TrafficLightState.Off:     'Off',
    carla.TrafficLightState.Unknown: 'Unknown',
}


def parse_args():
    p = argparse.ArgumentParser(description='CARLA HD-map downloader')
    p.add_argument('--host',      default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    p.add_argument('--interval',  default=2.0, type=float,
                   help='웨이포인트 샘플링 간격 (m, default: 2.0)')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _loc(obj):
    """carla.Location / Transform.location → dict"""
    if hasattr(obj, 'location'):
        obj = obj.location
    return {'x': round(obj.x, 3), 'y': round(obj.y, 3), 'z': round(obj.z, 3)}


def _rot(tf):
    r = tf.rotation
    return {
        'roll':  round(r.roll,  3),
        'pitch': round(r.pitch, 3),
        'yaw':   round(r.yaw,   3),
    }


def _wp_entry(wp):
    tf = wp.transform
    return {
        'road_id':    wp.road_id,
        'section_id': wp.section_id,
        'lane_id':    wp.lane_id,
        'junction_id': wp.junction_id if wp.is_junction else None,
        'is_junction': wp.is_junction,
        'lane_type':  _LANE_TYPE_STR.get(wp.lane_type, str(wp.lane_type)),
        'lane_width': round(wp.lane_width, 3),
        's':          round(wp.s, 3),
        'x':          round(tf.location.x, 3),
        'y':          round(tf.location.y, 3),
        'z':          round(tf.location.z, 3),
        'yaw':        round(tf.rotation.yaw, 3),
    }


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_waypoints(world_map, interval: float):
    wps = world_map.generate_waypoints(interval)
    return [_wp_entry(wp) for wp in wps]


def extract_lanes(world_map, interval: float):
    """도로별 차선 중심선 폴리라인."""
    lanes: dict[tuple, list] = {}
    for wp in world_map.generate_waypoints(interval):
        key = (wp.road_id, wp.section_id, wp.lane_id)
        if key not in lanes:
            lanes[key] = {
                'road_id':    wp.road_id,
                'section_id': wp.section_id,
                'lane_id':    wp.lane_id,
                'lane_type':  _LANE_TYPE_STR.get(wp.lane_type, str(wp.lane_type)),
                'lane_width': round(wp.lane_width, 3),
                'centerline': [],
            }
        tf = wp.transform
        lanes[key]['centerline'].append({
            's': round(wp.s, 3),
            'x': round(tf.location.x, 3),
            'y': round(tf.location.y, 3),
            'z': round(tf.location.z, 3),
            'yaw': round(tf.rotation.yaw, 3),
        })
    return list(lanes.values())


def extract_junctions(world_map):
    topology = world_map.get_topology()
    seen = set()
    junctions = []
    for wp_a, wp_b in topology:
        for wp in (wp_a, wp_b):
            if wp.is_junction and wp.junction_id not in seen:
                seen.add(wp.junction_id)
                try:
                    junc = wp.get_junction()
                    wps_in = junc.get_waypoints(carla.LaneType.Driving)
                    entries = [{'entry': _wp_entry(s), 'exit': _wp_entry(e)} for s, e in wps_in]
                    center = junc.bounding_box.location
                    ext    = junc.bounding_box.extent
                    junctions.append({
                        'junction_id': wp.junction_id,
                        'center': {'x': round(center.x, 3),
                                   'y': round(center.y, 3),
                                   'z': round(center.z, 3)},
                        'extent': {'x': round(ext.x, 3),
                                   'y': round(ext.y, 3),
                                   'z': round(ext.z, 3)},
                        'paths': entries,
                    })
                except Exception:
                    junctions.append({'junction_id': wp.junction_id})
    return junctions


def extract_traffic_lights(world):
    """신호등마다 실제 정지선 waypoint(get_stop_waypoints)도 함께 저장한다.

    학습 파이프라인(data/src/map_loader.py LaneMap.ego_traffic_light_stop)이 ego의
    현재 차선(road_id/section_id/lane_id)과 stop_waypoints의 차선을 매칭해, "ego 차선에
    실제로 연결된 신호등"만 골라내고 그 신호등의 실제 정지선 좌표를 쓴다 — pole 위치나
    거리 기반 근사가 아니라 vehicle.get_traffic_light()와 동등한 효과를 정적 맵 데이터로
    재현하는 방식이라, 이미 녹화된 세션에도 재-실행 없이 소급 적용된다.
    """
    tls = []
    for tl in world.get_actors().filter('traffic.traffic_light'):
        loc = tl.get_location()
        stop_waypoints = []
        for wp in tl.get_stop_waypoints():
            wp_loc = wp.transform.location
            stop_waypoints.append({
                'road_id':    wp.road_id,
                'section_id': wp.section_id,
                'lane_id':    wp.lane_id,
                'x':          round(wp_loc.x, 3),
                'y':          round(wp_loc.y, 3),
            })
        tls.append({
            'id':      tl.id,
            'x':       round(loc.x, 3),
            'y':       round(loc.y, 3),
            'z':       round(loc.z, 3),
            'yaw':     round(tl.get_transform().rotation.yaw, 3),
            'state':   _TL_STATE_STR.get(tl.get_state(), 'Unknown'),
            'red_t':   round(tl.get_red_time(),    2),
            'yellow_t': round(tl.get_yellow_time(), 2),
            'green_t':  round(tl.get_green_time(),  2),
            'stop_waypoints': stop_waypoints,
        })
    return tls


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world     = client.get_world()
    world_map = world.get_map()
    map_name  = world_map.name.split('/')[-1]  # e.g. "Town03"

    os.makedirs(_OUT_DIR, exist_ok=True)

    # ── 1. OpenDRIVE 원본 저장 ────────────────────────────────────────────
    xodr_path = os.path.join(_OUT_DIR, f'{map_name}.xodr')
    with open(xodr_path, 'w', encoding='utf-8') as f:
        f.write(world_map.to_opendrive())
    print(f'[map] OpenDRIVE 저장 → {xodr_path}')

    # ── 2. 레이어별 추출 ──────────────────────────────────────────────────
    print(f'[map] 웨이포인트 추출 (interval={args.interval} m) ...')
    waypoints = extract_waypoints(world_map, args.interval)
    print(f'      → {len(waypoints):,} waypoints')

    print('[map] 차선 중심선 추출 ...')
    lanes = extract_lanes(world_map, args.interval)
    print(f'      → {len(lanes):,} lanes')

    print('[map] 교차로 추출 ...')
    junctions = extract_junctions(world_map)
    print(f'      → {len(junctions):,} junctions')

    print('[map] 신호등 추출 ...')
    traffic_lights = extract_traffic_lights(world)
    print(f'      → {len(traffic_lights):,} traffic lights')

    # ── 3. JSON 저장 ──────────────────────────────────────────────────────
    payload = {
        'map_name':      map_name,
        'interval_m':    args.interval,
        'waypoints':     waypoints,
        'lanes':         lanes,
        'junctions':     junctions,
        'traffic_lights': traffic_lights,
    }
    json_path = os.path.join(_OUT_DIR, f'{map_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'[map] JSON 저장 → {json_path}')
    print('[map] Done.')


if __name__ == '__main__':
    main()

"""Carla 연동 공용 유틸 — hero 스폰/route 생성/received 정렬.

rule_based_test.py와 scenario_eval.py가 공유하는 순수 유틸만 모아 둔다(주행 로직 자체는
각자 파일에 있음). GlobalRoutePlanner 로드는 CARLA_ROOT 환경변수(기본 ~/carla)가
PythonAPI/carla/agents를 포함한 CARLA 설치 경로를 가리켜야 한다.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# est_pos(UWB+오도메트리 KF 융합 위치)가 route_wps에서 이 거리(m) 이상 벗어나면 "route 이탈"로
# 보고 현재 위치 기준으로 route를 통째로 재생성한다. 차선폭(~3.5m)이나 커브 구간에서 route
# polyline 보간이 주는 정상적인 편차보다는 확실히 크지만, est_pos 초기 미수렴/drift로 route가
# 실제 위치와 수십~수백 m 어긋나는 경우는 확실히 잡을 만큼 작은 값.
ROUTE_RESYNC_DIST_M = 15.0


def _dist(ego_x: float, ego_y: float, p: dict[str, Any]) -> float:
    dx = (p.get("x") or 0.0) - ego_x
    dy = (p.get("y") or 0.0) - ego_y
    return math.hypot(dx, dy)


def _sort_received(raw: dict[Any, dict[str, Any]], ego_x: float, ego_y: float) -> dict[str, list[dict]]:
    """각 payload에 ego 기준 dist_m을 채우고 종류별(vehicle/pedestrian/infrastructure)로
    가까운 순 정렬 — rule_based_control이 코리도 판정 시 가까운 대상부터 보게 한다."""
    infra_types = {"anchor", "traffic_light"}
    vehicles, pedestrians, infrastructure = [], [], []
    for payload in raw.values():
        t = payload.get("type")
        entry = dict(payload)
        entry["dist_m"] = _dist(ego_x, ego_y, payload)
        if t == "vehicle":
            vehicles.append(entry)
        elif t == "pedestrian":
            pedestrians.append(entry)
        elif t in infra_types:
            infrastructure.append(entry)
    vehicles.sort(key=lambda e: e["dist_m"])
    pedestrians.sort(key=lambda e: e["dist_m"])
    infrastructure.sort(key=lambda e: e["dist_m"])
    return {"vehicles": vehicles, "pedestrians": pedestrians, "infrastructure": infrastructure}


def _load_route_planner(sampling_resolution: float = 2.0):
    """CARLA PythonAPI의 GlobalRoutePlanner 로드.

    pip 패키지 carla에는 안 들어있고 CARLA 설치 폴더의 PythonAPI/carla 아래에만 있어서,
    CARLA_ROOT 환경변수(기본 ~/carla)로 위치를 찾는다.
    """
    carla_root = Path(os.environ.get("CARLA_ROOT", Path.home() / "carla"))
    agents_path = carla_root / "PythonAPI" / "carla"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"CARLA agents 모듈을 찾을 수 없습니다 ({agents_path}). "
            "CARLA_ROOT 환경변수로 CARLA 설치 경로를 지정하세요 (PythonAPI/carla/agents가 있는 폴더)."
        ) from exc

    def build(world) -> "GlobalRoutePlanner":
        return GlobalRoutePlanner(world.get_map(), sampling_resolution)

    return build


def _route_length(route_wps: list[tuple[float, float]], x: float, y: float) -> float:
    """route_wps를 따라 (x, y)에서 마지막 waypoint까지 남은 길이(m).

    ego가 route_wps[0]에 정확히 있지 않을 수 있어(투영점과 실제 위치 사이 간격) (x, y)에서
    route_wps[0]까지의 거리를 첫 구간으로 포함해서 계산한다.
    """
    if not route_wps:
        return 0.0
    total = math.hypot(route_wps[0][0] - x, route_wps[0][1] - y)
    for i in range(len(route_wps) - 1):
        ax, ay = route_wps[i]
        bx, by = route_wps[i + 1]
        total += math.hypot(bx - ax, by - ay)
    return total


# route 앞부분에서 한 waypoint→다음 waypoint로 넘어갈 때 진행 방향이 이 각도(도)보다
# 크게 꺾이면 차가 실제로 낼 수 없는 급반전으로 본다. 2m 간격 waypoint에서 가장 급한
# 교차로 좌/우회전도 한 스텝당 ~30~45°만 꺾이므로(반경이 작아도) 이 값을 넘지 않는다.
# 반대로 GRP가 교차로 connector를 잘못 이어 "위로 갔다 되돌아오는" 갈고리(hook)를 만들면
# 그 꼭짓점에서 방향이 90°를 훌쩍 넘겨 뒤집히므로(실측 ~120~150°) 이 기준에만 걸린다.
ROUTE_MAX_TURN_DEG = 90.0


def _sharpest_turn_deg(points: list[tuple[float, float]], upto: int = 25) -> float:
    """route 폴리라인 앞부분(첫 upto segment)에서 한 waypoint에서 다음으로 넘어갈 때
    진행 방향이 꺾이는 최대 각도(도). 교차로에서 route가 물리적으로 불가능하게 급반전하는
    (차선을 무시하고 위로 갔다 되돌아오는) 갈고리를 감지하는 데 쓴다."""
    worst = 0.0
    prev_ang: float | None = None
    seg = points[: upto + 1]
    for i in range(len(seg) - 1):
        dx = seg[i + 1][0] - seg[i][0]
        dy = seg[i + 1][1] - seg[i][1]
        if dx * dx + dy * dy < 1e-6:
            continue  # 중복 waypoint는 방향이 정의 안 되므로 건너뜀
        ang = math.atan2(dy, dx)
        if prev_ang is not None:
            d = abs(math.degrees(ang - prev_ang)) % 360.0
            worst = max(worst, min(d, 360.0 - d))
        prev_ang = ang
    return worst


def _dehook_route(
    points: list[tuple[float, float]], max_turn_deg: float = ROUTE_MAX_TURN_DEG
) -> list[tuple[float, float]]:
    """route 폴리라인 **전체**를 훑어, 한 waypoint에서 진행 방향이 max_turn_deg보다 크게
    뒤집히는 갈고리(hook) 꼭짓점을 제거한다.

    _sharpest_turn_deg 검사는 route 앞부분만 보므로, GRP가 route 뒤쪽에 만든 갈고리는
    걸러지지 않고 통과한다 — 그 상태로 주행하면 앞 waypoint가 소비되며 뒤쪽 갈고리가 앞으로
    나와, route가 갑자기 꺾이고 차가 그쪽으로 조향한다. 정상 교차로 회전은 waypoint당
    각도가 작아(2~5m 간격에서 <45°) 남고, 물리적으로 불가능한 반전 꼭짓점만 지워 이웃 두
    점을 직접 잇는다. 여러 점이 연속으로 튀는 갈고리도 처리하도록 변화가 없을 때까지
    반복한다. 시작/끝 점은 항상 보존한다(경로의 실제 출발/도착)."""
    if len(points) < 3:
        return list(points)
    pts = list(points)
    while len(pts) >= 3:
        out = [pts[0]]
        removed = False
        for i in range(1, len(pts) - 1):
            ax, ay = out[-1]
            bx, by = pts[i]
            cx, cy = pts[i + 1]
            d1x, d1y = bx - ax, by - ay
            d2x, d2y = cx - bx, cy - by
            if d1x * d1x + d1y * d1y < 1e-6 or d2x * d2x + d2y * d2y < 1e-6:
                out.append(pts[i])
                continue
            turn = abs(math.degrees(math.atan2(d2y, d2x) - math.atan2(d1y, d1x))) % 360.0
            turn = min(turn, 360.0 - turn)
            if turn > max_turn_deg:
                removed = True  # 꼭짓점 pts[i] 버림 — out에 넣지 않음
            else:
                out.append(pts[i])
        out.append(pts[-1])
        pts = out
        if not removed:
            break
    return pts


def _trim_to_forward(
    points: list[tuple[float, float]], origin, yaw_deg: float
) -> list[tuple[float, float]]:
    """GlobalRoutePlanner.trace_route()는 origin이 속한 도로 구간의 뒤쪽(지나온 방향)
    waypoint까지 포함할 수 있어 route 첫 지점이 실제 위치보다 뒤이거나 진행 방향과
    반대편(behind)일 수 있다. yaw_deg(진행 방향) 기준으로 내 앞(forward>=0)에 있는 첫
    waypoint부터 시작하도록 자른다. 앞에 있는 점을 하나도 못 찾으면(추정 위치 오차 등)
    0으로 되돌아가 전체 경로를 그대로 쓴다 — 목적지 한 점만 남기면 도로를 무시하고
    직선으로 잇는 "물리적으로 불가능한" route가 나오기 때문."""
    if len(points) <= 1:
        return points
    yaw = math.radians(yaw_deg)
    fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
    start_idx = 0
    for i, (px, py) in enumerate(points):
        if (px - origin.x) * fwd_x + (py - origin.y) * fwd_y >= 0.0:
            start_idx = i
            break
    return points[start_idx:]


# _prune_pingpong_lane_changes에서, 차선변경 시작 지점 이후 이 거리(m) 안에 원래 차선
# (road_id/lane_id 동일)으로 돌아오면 "목적 없는" 차선변경으로 본다. GRP가 만드는 실제
# 차선변경은 waypoint 몇 개(약 10~20m) 안에 끝나므로, 그보다 넉넉히 잡아 정상적인 차선변경
# 완료 구간은 놓치지 않으면서 되돌아오는 패턴은 확실히 잡는다.
LANE_CHANGE_PINGPONG_MAX_M = 30.0


def _prune_pingpong_lane_changes(
    route: list[tuple[Any, Any]], max_span_m: float = LANE_CHANGE_PINGPONG_MAX_M,
) -> list[tuple[Any, Any]]:
    """trace_route()가 만드는 "옆 차선으로 갔다가 바로 되돌아오는" 무의미한 차선변경을 제거.

    GlobalRoutePlanner는 순수 최단 topology 경로만 찾기 때문에, 목적지 도달에 꼭 필요하지
    않은데도 CHANGELANELEFT/RIGHT로 옆 차선에 들어갔다가 max_span_m 안에서 같은
    road_id/lane_id로 되돌아오는 waypoint 시퀀스를 낼 수 있다. 차선변경 시작 지점(원래
    차선의 마지막 waypoint)과, 같은 차선으로 돌아온 지점을 직선으로 바로 이어 그 사이
    waypoint를 버린다 — 두 지점 다 원래 차선 위이므로 지름길로 이어도 차선을 벗어나지
    않는다."""
    if len(route) < 3:
        return route
    road_option = type(route[0][1])
    lane_change_opts = {road_option.CHANGELANELEFT, road_option.CHANGELANERIGHT}
    out: list[tuple[Any, Any]] = []
    i = 0
    n = len(route)
    while i < n:
        wp, opt = route[i]
        if opt in lane_change_opts and out:
            start_wp = out[-1][0]
            found = -1
            j = i + 1
            while j < n:
                cand_wp = route[j][0]
                if start_wp.transform.location.distance(cand_wp.transform.location) > max_span_m:
                    break
                if cand_wp.road_id == start_wp.road_id and cand_wp.lane_id == start_wp.lane_id:
                    found = j
                    break
                j += 1
            if found >= 0:
                i = found
                continue
        out.append(route[i])
        i += 1
    return out


def _pick_new_route(
    world, grp, origin, yaw_deg: float, min_dist: float = 30.0,
    max_turn_deg: float = ROUTE_MAX_TURN_DEG, tries: int = 8,
) -> tuple[list[tuple[float, float]], "carla.Location"]:
    """origin(현재 위치)에서 충분히 먼 랜덤 spawn point까지의 route를 생성.

    GlobalRoutePlanner.trace_route()가 교차로(특히 평행 차선이 여러 겹 쌓인 junction)에서
    connector를 잘못 이어 "위로 갔다 되돌아오는" 물리적으로 불가능한 갈고리 route를 내는
    경우가 있다. 그런 route는 waypoint를 지워 고칠 수 없다(지우면 차선을 가로지르는 직선
    지름길이 됨) — 애초에 그 목적지를 버리고 다른 랜덤 목적지로 route를 다시 뽑는다.
    앞부분(route_ahead가 되는 구간)에 급반전(> max_turn_deg)이 없는 첫 route를 채택하고,
    tries번 안에 못 찾으면 그중 가장 완만한 route를 쓴다(빈 route보다 나음)."""
    spawn_points = world.get_map().get_spawn_points()
    candidates = [sp for sp in spawn_points if sp.location.distance(origin) > min_dist] or spawn_points
    best: tuple[list[tuple[float, float]], "carla.Location", float] | None = None
    for _ in range(tries):
        dest = random.choice(candidates)
        route = grp.trace_route(origin, dest.location)
        route = _prune_pingpong_lane_changes(route)
        points = _trim_to_forward(
            [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route], origin, yaw_deg,
        )
        # route 전체의 갈고리(hook)를 먼저 제거해, 뒤쪽 갈고리가 나중에 앞으로 나와 차가
        # 갑자기 꺾이는 것을 막는다. 그 뒤 앞부분 급반전을 한 번 더 확인한다.
        points = _dehook_route(points)
        if len(points) < 2:
            continue
        turn = _sharpest_turn_deg(points)
        if turn <= max_turn_deg:
            return points, dest.location
        if best is None or turn < best[2]:
            best = (points, dest.location, turn)
    if best is not None:
        return best[0], best[1]
    return [], origin  # 모든 후보가 2점 미만 — 호출부에서 "route 생성/연장 실패"로 처리


def _find_or_spawn_hero(world, role_name: str = "hero", random_spawn: bool = False):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            return actor, False

    bp_lib = world.get_blueprint_library()
    candidates = bp_lib.filter("vehicle.tesla.model3") or bp_lib.filter("vehicle.*")
    bp = candidates[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    spawn_points = world.get_map().get_spawn_points()
    if random_spawn:
        spawn_points = list(spawn_points)
        random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        actor = world.try_spawn_actor(bp, spawn_point)
        if actor is not None:
            return actor, True
    raise RuntimeError("No free spawn point for hero vehicle — clear traffic or pick another map.")

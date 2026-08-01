"""Carla 시뮬레이터 연동 — 규칙 기반(rule-based) 로직으로 hero 차량을 주행.

Carla 서버에 접속해 hero를 찾거나 스폰하고, route를 GlobalRoutePlanner로 생성/연장하며,
제어값은 아래 규칙으로 계산한다.
  - 조향: route waypoint를 따라가는 순수 추종(pure pursuit)
  - 속도: 기본 순항 속도(CRUISE_SPEED_MPS)를 유지하되, route 코리도 안에 있는 빨간/노란
    신호등이나 선행 차량·보행자가 있으면 속도 상한을 걸어 감속/정지. 신호등/보행자는
    정지거리 공식(v = sqrt(2*a*d)), 선행 차량은 그 차의 속도를 반영한 ACC 공식
    (v = sqrt(v_lead^2 + 2*a*d))으로 상한을 계산해 앞차 속도에 부드럽게 수렴한다.
Carla 연동 준비 절차(CARLA_ROOT 환경변수 등)는 test/carla_common.py 상단 docstring 참고.

실행: python test/rule_based_test.py
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from data.src.control import SpeedPIController  # noqa: E402
from data.src.map_loader import LaneMap, get_lane_map  # noqa: E402
from data.src.paths import MAP_JSON  # noqa: E402
from data.src.polyline import SIGNAL_MAP, world_to_ego  # noqa: E402

# route 생성(_pick_new_route)·길이(_route_length)·정렬(_sort_received)·스폰(_find_or_spawn_hero)
# 등 순수 유틸은 test/carla_common.py에 공용으로 모아두고 재사용한다. route 소비
# (_advance_route)와 route 유지 로직은 "하나의 route에 커밋 후, 남은 길이가 입력 route
# 길이 이하일 때만 연장"하도록 여기서 새로 정의한다.
from carla_common import (  # noqa: E402
    ROUTE_RESYNC_DIST_M,
    _find_or_spawn_hero,
    _load_route_planner,
    _pick_new_route,
    _route_length,
    _sort_received,
)

# 전방이 비어 있을 때 유지할 순항 속도. (20 km/h)
CRUISE_SPEED_MPS = 20.0 / 3.6

# route waypoint 간격(m)과 lookahead 개수 — carla_test.py(모델 입력, 2m*25=50m)와는 별개로
# 여기서는 20개*5m=100m로 더 길게 잡는다. 신호를 아직 못 본 상태에서 속도가 빠르면 route가
# 짧을 때 route 끝 근처에서 pure pursuit lookahead가 남은 waypoint를 못 잡아 조향이
# 튀는(route가 이상해 보이는) 문제가 있어, lookahead 자체를 넉넉하게 늘려서 완화한다.
ROUTE_SAMPLING_RESOLUTION_M = 5.0
ROUTE_LOOKAHEAD_POINTS = 20

# 아래 정지거리 임계값(m)은 원래 train/loss.py의 _unnecessary_stop_loss·_red_light_loss가
# "정지해도 되는(should_stop) 상황"으로 보는 기준과 맞춘 것이었으나, 실주행에서 차량/신호/보행자
# 충돌이 남아 있어 안전 쪽으로 강화했다(모델 학습 기준보다 보수적) — 코멘트 참고.
# 코리도 '폭' 판정 기하도 실주행 안전을 위해 loss보다 강화했다: 앞차는 ego 정면 직선축이
# 아니라 route 폴리라인 기준 측방거리 + 앞차 실제 폭(width/2)으로 감지한다(_corridor_ahead 참고).
STOP_VEHICLE_DIST_M = 6.0
# 보행자는 UWB 페이로드에 속도가 없어(agent/pedestrian.py 참고) 차량과 달리 진행방향 예측이
# 불가능하다 — 지금 코리도 안에 있어야만 감지되므로, 코리도 폭과 정지 여유를 더 넉넉히 잡아
# "코리도 밖에서 갑자기 들어오는" 보행자에도 반응할 시간을 번다.
STOP_PED_DIST_M = 8.0
STOP_LIGHT_DIST_M = 3.0
LANE_HALF_WIDTH_M = 1.3
PED_LANE_HALF_WIDTH_M = 2.5
# 선행 차량 정지 여유(범퍼 간격, m) — 값이 클수록 앞차와의 간격을 넉넉히 두고 감속을 일찍 시작.
VEHICLE_GAP_M = 6.0

# 감속도(m/s^2) 두 가지를 분리한다:
#   - SPEED_LIMIT_DECEL_MSS: 정지거리 역산(v = sqrt(2*a*d))에서 "이 정도 감속이면 충분하다"고
#     가정하는 보수적인 계획값. 실제 제동 능력보다 낮게 잡아, 반응 지연·센서 노이즈·제어 랙이
#     있어도 margin을 다 쓰기 전에 이미 감속이 끝나 있도록 여유를 만든다.
#   - MAX_DECEL_MSS: 속도 PI 제어기가 실제로 낼 수 있는 최대 감속 상한(=brake=1.0에 대응).
#     계획값보다 높게 잡아, 예상보다 늦게 감지되거나(보행자 등) 목표가 급히 낮아져도 그 차이를
#     따라잡을 제동 여력을 남겨 둔다.
SPEED_LIMIT_DECEL_MSS = 2.5
MAX_DECEL_MSS = 5.0

# 목표 속도가 이 값(m/s) 이하로 눌리면 사실상 "정지 명령"으로 보고 정지 이유를 판정한다.
STOP_REPORT_SPEED_MPS = 0.5

# pure pursuit 조향에 쓰는 lookahead 거리(m) 계산 파라미터 및 차량 축거.
MIN_LOOKAHEAD_M = 3.0
LOOKAHEAD_GAIN = 1.2
WHEELBASE_M = 2.85
MAX_STEER_RAD = 0.9


def _ts() -> str:
    """로그 접두사에 붙일 현재 시각 (HH:MM:SS.mmm)."""
    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


# 목적지까지 하나의 route에 '커밋'한 뒤, 남은 길이가 이 값(=입력으로 나가는 route_ahead 길이,
# ROUTE_LOOKAHEAD_POINTS*ROUTE_SAMPLING_RESOLUTION_M) 이하로 줄었을 때에만 목적지를 새로
# 이어 붙인다. 중간에 route를 다시 뽑지 않으므로 "route가 갑자기 바뀌는" 일이 사라진다.
INPUT_ROUTE_LENGTH_M = ROUTE_LOOKAHEAD_POINTS * ROUTE_SAMPLING_RESOLUTION_M


def _advance_route(
    route_wps: list[tuple[float, float]], x: float, y: float, max_advance_m: float = 40.0,
) -> tuple[list[tuple[float, float]], float]:
    """ego 위치(x, y)를 route_wps 폴리라인에 투영해서 이미 지나간 구간의 waypoint를 제거.

    투영은 route 앞쪽 max_advance_m 구간 안에서만 한다. 목적지를 계속 이어 붙이면 경로가
    자기 근처로 되돌아오는 self-crossing이 생기는데, route 전체를 투영 대상으로 삼으면 저
    멀리 앞에서 다시 가까워지는 구간이 최단점으로 뽑혀 ego가 그쪽으로 순간 점프하고 route가
    "갑자기 바뀐" 것처럼 보인다. ego는 한 tick에 몇 m만 움직이므로 앞쪽 수십 m 안에서만 최단
    segment를 찾으면 진행이 항상 국소·단조적이라 이 점프가 사라진다.

    반환 두 번째 값은 (x, y)에서 route까지 최단거리(m) — route가 실제 위치와 완전히
    동떨어진 경우(est_pos 미수렴/drift)를 호출부가 감지하는 데 쓴다."""
    if len(route_wps) < 2:
        offset = math.hypot(route_wps[0][0] - x, route_wps[0][1] - y) if route_wps else math.inf
        return route_wps, offset
    best_idx = 0
    best_dist_sq = math.inf
    cum_len = 0.0
    for i in range(len(route_wps) - 1):
        ax, ay = route_wps[i]
        bx, by = route_wps[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        t = 0.0 if seg_len_sq < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
        px, py = ax + t * dx, ay + t * dy
        dist_sq = (px - x) ** 2 + (py - y) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_idx = i + 1 if t >= 1.0 else i
        cum_len += math.sqrt(seg_len_sq)
        if cum_len > max_advance_m:
            break
    return route_wps[best_idx:], math.sqrt(best_dist_sq)


def _maintain_route(
    world, grp, carla_mod, route_wps: list[tuple[float, float]], route_dest,
    x: float, y: float, yaw: float, off_route_ticks: int,
    *, route_dest_min_dist: float, input_route_length_m: float,
    resync_dist_m: float, resync_persist_ticks: int,
) -> tuple[list[tuple[float, float]], Any, int, list[str]]:
    """route를 '하나에 커밋 후, 남은 길이가 입력 route 길이 이하일 때만 연장'하는 방식으로 유지.

    반환: (route_wps, route_dest, off_route_ticks, events)
      - events: 상황이 바뀔 때만 담기는 로그 문자열 리스트(호출부가 접두사 붙여 출력).

    1) _advance_route로 지나간 구간 제거(앞쪽 국소 구간만 투영 → self-crossing 점프 방지).
    2) est_pos가 커밋된 route에서 resync_dist_m 이상 벗어난 상태가 resync_persist_ticks tick
       '연속'으로 지속될 때만 route를 폐기·재생성(한 tick 노이즈로는 route를 버리지 않는다).
    3) route가 없으면 목적지까지 새 route 생성(커밋).
    4) 남은 길이가 input_route_length_m 이하로 줄면 현재 목적지에서 다른 목적지까지 route를
       구해 이어 붙인다. 그 외에는 커밋된 route를 그대로 따른다(중간에 재생성 안 함)."""
    events: list[str] = []
    route_wps, route_offset_m = _advance_route(route_wps, x, y)

    if route_wps and route_dest is not None and route_offset_m > resync_dist_m:
        off_route_ticks += 1
        if off_route_ticks >= resync_persist_ticks:
            events.append(
                f"route 이탈 (ego-route {route_offset_m:.1f}m > {resync_dist_m:.1f}m, "
                f"{off_route_ticks} tick 연속) → 현재 위치 기준 route 재생성"
            )
            route_wps, route_dest, off_route_ticks = [], None, 0
    else:
        off_route_ticks = 0

    if route_dest is None:
        route_wps, route_dest = _pick_new_route(
            world, grp, carla_mod.Location(x=x, y=y), yaw, min_dist=route_dest_min_dist,
        )
        events.append(f"목적지까지 route {len(route_wps)} waypoints 생성(커밋)")
        return route_wps, route_dest, off_route_ticks, events

    if _route_length(route_wps, x, y) <= input_route_length_m or len(route_wps) <= 1:
        if len(route_wps) >= 2:
            (ex, ey), (lx, ly) = route_wps[-2], route_wps[-1]
            end_yaw = math.degrees(math.atan2(ly - ey, lx - ex))
        else:
            end_yaw = yaw
        extension, new_dest = _pick_new_route(
            world, grp, route_dest, end_yaw, min_dist=route_dest_min_dist,
        )
        if extension:
            if route_wps and math.hypot(
                extension[0][0] - route_wps[-1][0], extension[0][1] - route_wps[-1][1]
            ) < 0.5:
                extension = extension[1:]
            route_wps = route_wps + extension
            route_dest = new_dest
            events.append(
                f"route 연장 +{len(extension)} waypoints "
                f"(총 {len(route_wps)}, 남은 {_route_length(route_wps, x, y):.1f}m)"
            )
        else:
            events.append("route 연장 실패 (trace_route 빈 결과) — 다음 tick 재시도")
    return route_wps, route_dest, off_route_ticks, events


def _route_ego_ahead(
    route_wps: list[tuple[float, float]], x: float, y: float, yaw_deg: float,
) -> list[tuple[float, float]]:
    """route_wps(world 좌표)를 ego-local(전방=x, 우측=y)로 변환해 전방(x>0) 점만 반환."""
    ahead = []
    for wx, wy in route_wps:
        lx, ly = world_to_ego(wx, wy, x, y, yaw_deg)
        if lx > 0:
            ahead.append((lx, ly))
    return ahead


def _pure_pursuit_steer(route_ego: list[tuple[float, float]], current_speed: float) -> float:
    """route_ego(전방 ego-local 점들)를 따라가는 순수 추종 조향각(-1~1)."""
    if not route_ego:
        return 0.0
    lookahead = max(MIN_LOOKAHEAD_M, LOOKAHEAD_GAIN * max(current_speed, 0.0))
    lx, ly = route_ego[-1]
    for px, py in route_ego:
        if math.hypot(px, py) >= lookahead:
            lx, ly = px, py
            break
    ld = math.hypot(lx, ly)
    if ld < 1e-3:
        return 0.0
    alpha = math.atan2(ly, lx)
    curvature = 2.0 * math.sin(alpha) / ld
    steer_rad = math.atan(WHEELBASE_M * curvature)
    return float(max(-1.0, min(1.0, steer_rad / MAX_STEER_RAD)))


def _dist_to_polyline(px: float, py: float, poly: list[tuple[float, float]]) -> float:
    """점 (px,py)에서 폴리라인 poly까지의 최단(수직) 거리. poly가 비면 원점까지 거리로 대체."""
    if not poly:
        return math.hypot(px, py)
    if len(poly) == 1:
        return math.hypot(px - poly[0][0], py - poly[0][1])
    best = math.inf
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        best = min(best, d)
    return best


def _project_route(
    px: float, py: float, route_ego: list[tuple[float, float]],
) -> tuple[float, float]:
    """점 (px,py)를 route_ego 폴리라인에 투영 → (arc_len, lateral).

    - arc_len: route 시작점부터 투영점까지 경로를 따라간 거리(m). 직선 전방거리 lx가 아니라
      '길을 따라' 잰 거리라, 커브 구간의 같은 차선 앞차/정지선까지 거리를 제대로 준다.
    - lateral: 투영점까지 수직거리(m) — 코리도(같은 차선/경로) 판정용.
    route_ego가 비면 원점 기준 거리로 대체한다."""
    if not route_ego:
        d = math.hypot(px, py)
        return d, d
    if len(route_ego) == 1:
        d = math.hypot(px - route_ego[0][0], py - route_ego[0][1])
        return d, d
    best_lat = math.inf
    best_arc = 0.0
    cum = 0.0
    for (ax, ay), (bx, by) in zip(route_ego, route_ego[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            t, foot_x, foot_y = 0.0, ax, ay
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            foot_x, foot_y = ax + t * dx, ay + t * dy
        lat = math.hypot(px - foot_x, py - foot_y)
        if lat < best_lat:
            best_lat = lat
            best_arc = cum + t * math.sqrt(seg2)
        cum += math.sqrt(seg2)
    return best_arc, best_lat


def _corridor_candidates(
    lx: float, ly: float, route_ego: list[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """(lx,ly)의 코리도 판정 후보 (lateral, 대표거리) 목록.

    ego 정면 직진축(|ly|, 직선거리 lx) 기준은 route 유무와 무관하게 항상 포함하고,
    route_ego가 있으면 route 폴리라인 기준(수직거리, arc 거리)도 추가한다 — 두 기준을
    OR로 합쳐서, route가 커브로 휘어 코리도 밖으로 벗어난 지점이라도 ego 정면에 실제로
    있는 대상은 놓치지 않게 한다(반대로 route만으로 잡히는, 직진축 밖이지만 커브를 따라가면
    코리도 안인 대상도 그대로 유지). 호출부는 lateral<=half_width인 후보들 중 대표거리가
    가장 작은(=가장 보수적인) 것을 쓴다."""
    candidates = [(abs(ly), lx)]
    if route_ego:
        arc, lateral = _project_route(lx, ly, route_ego)
        candidates.append((lateral, arc))
    return candidates


def _corridor_ahead(
    entries: list[dict[str, Any]], x: float, y: float, yaw_deg: float, half_width_m: float,
    route_ego: list[tuple[float, float]] | None = None, use_actor_width: bool = False,
) -> tuple[float, dict[str, Any]] | tuple[None, None]:
    """entries 중 ego 전방(x>0)이고 진행 경로 코리도 안에 있는 가장 가까운 대상의
    (전방거리 m, entry). 해당하는 entry가 없으면 (None, None).

    측방 판정 기준: ego 정면 직진축과 route 코리도를 _corridor_candidates로 함께 보고
    (OR), 어느 한쪽이라도 half_width 안이면 코리도 안으로 잡는다. 신호등 앞 대기열처럼
    커브/교차로 진입부에서 직진축만으로는 놓치는 경우를 route가 보완하고, 반대로 route가
    노이즈로 살짝 어긋나거나 없을 때는 직진축이 놓치지 않는 안전망이 된다.
      - use_actor_width면 상대의 실제 폭(width)의 절반을 코리도에 더한다. UWB 좌표는 상대 '중심'
        이라, 폭이 넓은 차는 중심이 코리도 밖이어도 차체가 내 경로를 물 수 있어서다.

    entry에는 상대 차량 치수('depth' 등)가 들어있어 범퍼 기준 정지거리 계산에 쓴다.
    """
    best_dist: float | None = None
    best_entry: dict[str, Any] | None = None
    for e in entries:
        ex, ey = e.get("x"), e.get("y")
        if ex is None or ey is None:
            continue
        lx, ly = world_to_ego(float(ex), float(ey), x, y, yaw_deg)
        if lx <= 0:
            continue
        hw = half_width_m
        if use_actor_width:
            w = e.get("width")
            if w:
                hw += float(w) / 2.0
        dists = [d for lateral, d in _corridor_candidates(lx, ly, route_ego) if lateral <= hw]
        if not dists:
            continue
        cand_dist = min(dists)
        if best_dist is None or cand_dist < best_dist:
            best_dist, best_entry = cand_dist, e
    if best_dist is None:
        return None, None
    return best_dist, best_entry


# route 기반 정지선 탐색에서 "route가 지나가는 정지선"으로 볼 측방 반폭(m). 차선 폭(1.3m)
# 그대로 쓰면 교차로 진입부의 거친 route 보간(5m 샘플링)이나 est_pos 노이즈로 실제로는
# 지나가는 정지선인데도 살짝 밖으로 밀려나 놓치는 사례가 있어, 인접 차선까지 여유 있게
# 잡히도록 한 차선 반 폭 정도(2.0m)로 넓힌다 — 정지 "거리" 기준(STOP_LIGHT_DIST_M 등)은
# 그대로라 더 멀리서 서는 일은 없고, "놓쳐서 못 서는" 쪽만 줄어든다.
LIGHT_ROUTE_HALF_WIDTH_M = 2.0

# ego_traffic_light_stop의 "내 위치와 가장 가까운 차선" 매칭 반경(m). 기본값(map_loader.py의
# 6.0m)은 est_pos 드리프트가 있으면 교차로 진입부에서 매칭 자체가 실패해(신호등을 아예 못
# 찾음) 놓치는 원인이 된다. 매칭 반경만 넓히는 것은 정지 거리에 영향을 주지 않으므로(정지선을
# "찾는" 범위이지 "얼마나 일찍 서는지"가 아님) 안전하게 늘릴 수 있다.
LIGHT_LANE_MATCH_RADIUS_M = 9.0


def _traffic_light_stop_dist(
    lane_map: LaneMap | None, infrastructure: list[dict[str, Any]], x: float, y: float, yaw_deg: float,
    route_ego: list[tuple[float, float]] | None = None,
    route_half_width_m: float = LIGHT_ROUTE_HALF_WIDTH_M,
    lane_match_radius_m: float = LIGHT_LANE_MATCH_RADIUS_M,
) -> float | None:
    """정지 대상(signal<=0.5) 신호등의 정지선까지 거리(m). 대상이 없으면 None.

    두 가지 방식을 함께 써서 "내가 있는 차선"에만 의존할 때 놓치는 케이스를 없앤다:
      (a) "내 위치(x,y)와 헤딩에 가장 가깝게 매칭되는 차선"에 연결된 신호등
          (lane_map.ego_traffic_light_stop — vehicle.get_traffic_light()와 동등한 정적 근사).
      (b) route_ego(전방 route 점들)가 주어지면, route가 실제로 지나가는 정지선들
          (lane_map.stop_by_lane_key 전체)도 훑어 route 코리도 안에 있는 것을 함께 본다.
          차선 경계 근처·헤딩이 아직 안정되지 않은 진입 구간 등 (a)가 지금 이 순간의 차선을
          잘못 매칭하거나 못 잡는 경우에도, route를 따라가면 곧 지나갈 정지선을 미리 잡는다.
    두 값 중 더 가까운(=더 보수적인) 거리를 쓴다 — 둘 중 하나라도 "정지해야 한다"고 하면
    반드시 그 거리 기준으로 정지하도록.

    signal<=0.5(SIGNAL_MAP: Red=0.0, Yellow=0.5, Green=1.0, Unknown=0.25)을 정지로 보는 기준은
    train/loss.py와 동일 — Unknown도 Green이 확인되기 전엔 정지 대상으로 취급한다."""
    if lane_map is None:
        return None

    signal_by_tl = {
        e.get("tl_id"): SIGNAL_MAP.get(e.get("signal", "Unknown"), 0.25)
        for e in infrastructure
        if e.get("type") == "traffic_light" and e.get("tl_id") is not None
    }

    dists: list[float] = []

    stop = lane_map.ego_traffic_light_stop(x, y, yaw_deg, radius_m=lane_match_radius_m)
    if stop is not None:
        tl_id, stop_x, stop_y = stop
        if signal_by_tl.get(tl_id, 1.0) <= 0.5:  # 초록이거나 신호 미보고면 정지 대상 아님
            lx, _ = world_to_ego(stop_x, stop_y, x, y, yaw_deg)
            if lx > 0:
                dists.append(lx)

    if route_ego:
        for tl_id, stop_x, stop_y in lane_map.stop_by_lane_key.values():
            if signal_by_tl.get(tl_id, 1.0) > 0.5:
                continue
            lx, ly = world_to_ego(stop_x, stop_y, x, y, yaw_deg)
            if lx <= 0:
                continue
            if _dist_to_polyline(lx, ly, route_ego) > route_half_width_m:
                continue
            dists.append(lx)

    return min(dists) if dists else None


VEHICLE_PREDICT_HORIZON_S = 2.0  # 교차/합류차 예측 지평(초)
VEHICLE_PREDICT_STEP_S = 0.4     # 예측 샘플 간격(초)


def _vehicle_hazard(
    vehicles: list[dict[str, Any]], x: float, y: float, yaw_deg: float,
    route_ego: list[tuple[float, float]] | None, half_width_m: float,
    predict_horizon_s: float = VEHICLE_PREDICT_HORIZON_S,
    predict_step_s: float = VEHICLE_PREDICT_STEP_S,
) -> tuple[float, dict[str, Any]] | tuple[None, None]:
    """내 경로(route)에 위협이 되는 가장 가까운 차량의 (경로 arc 거리 m, entry). 없으면 (None, None).

    두 가지를 함께 본다 — 그냥 정면만 보지 않고 같은 차선과 서로의 경로를 같이 살핀다:
      (a) 같은 차선/내 경로 위 선행차 — 지금 코리도 안(전방)에 있는 차. 코리도는
          _corridor_candidates로 ego 정면 직진축과 route를 OR로 함께 보고, route 기준으로
          잡혔으면 route를 따라간 arc 거리(커브 구간에서도 정확), 직진축 기준으로만 잡혔으면
          직선 전방거리를 쓴다 — route가 커브로 벗어나 있어도 정면에 있는 차를 놓치지 않는다.
      (b) 교차/합류차 — 지금은 코리도 밖이어도, 그 차의 속도로 predict_horizon_s초 앞을
          예측했을 때 내 경로 코리도로 들어오는 차. 상대의 진행 경로와 내 경로가 겹치는
          지점까지의 거리를 위협 거리로 본다. 멀리서 겹치는 교차는 _speed_limit_for_stop의
          정지거리 공식이 알아서 높은 속도를 허용하므로 자연히 무시되고, 가까울수록 강하게 감속한다.

    코리도 폭 hw = half_width_m + 상대 차량 폭/2 — UWB는 상대 '중심' 좌표라, 폭 넓은 차는
    중심이 코리도 밖이어도 차체가 내 경로를 물 수 있어 실제 폭을 더한다."""
    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    best_dist: float | None = None
    best_entry: dict[str, Any] | None = None
    for e in vehicles:
        ex, ey = e.get("x"), e.get("y")
        if ex is None or ey is None:
            continue
        lx, ly = world_to_ego(float(ex), float(ey), x, y, yaw_deg)
        hw = half_width_m
        w = e.get("width")
        if w:
            hw += float(w) / 2.0

        hazard_arc: float | None = None
        # (a) 지금 내 경로 코리도 안 & 전방 → 선행차
        if lx > 0:
            dists = [d for lateral, d in _corridor_candidates(lx, ly, route_ego) if lateral <= hw]
            if dists:
                hazard_arc = min(dists)

        # (b) 코리도 밖이면 그 차의 속도로 앞을 예측해 내 경로와 겹치는지 확인(교차/합류)
        if hazard_arc is None:
            vx_w = float(e.get("vel_x") or 0.0)  # 월드 프레임 속도
            vy_w = float(e.get("vel_y") or 0.0)
            v_fx = vx_w * cos_y + vy_w * sin_y   # ego 프레임으로 회전
            v_fy = -vx_w * sin_y + vy_w * cos_y
            if v_fx * v_fx + v_fy * v_fy > 0.25:  # 거의 정지한 옆 차량(<0.5m/s)은 예측 무의미
                t = predict_step_s
                while t <= predict_horizon_s:
                    px, py = lx + v_fx * t, ly + v_fy * t
                    if px > 0:
                        dists = [d for lateral, d in _corridor_candidates(px, py, route_ego) if lateral <= hw]
                        if dists:
                            hazard_arc = min(dists)
                            break
                    t += predict_step_s

        if hazard_arc is not None and (best_dist is None or hazard_arc < best_dist):
            best_dist, best_entry = hazard_arc, e

    if best_dist is None:
        return None, None
    return best_dist, best_entry


def _speed_limit_for_stop(
    dist: float | None, stop_margin: float, max_decel: float, lead_speed: float = 0.0,
) -> float:
    """dist(m) 앞의 대상까지 안전 간격(stop_margin)을 두고 접근하기 위한 목표 속도 상한.

    등가속 추종(ACC) 공식 v = sqrt(v_lead^2 + 2*a*remaining):
      - lead_speed=0(정지 신호등/보행자/멈춘 앞차)이면 v = sqrt(2*a*remaining) — 그 지점에서
        완전히 멈추는 기존 정지거리 공식과 동일.
      - lead_speed>0(움직이는 앞차)이면 그 속도까지만 줄이면 되므로 더 높은 속도를 허용해,
        앞차 속도에 부드럽게 수렴한다(불필요한 급감속/재가속 방지).
    remaining(=dist-stop_margin)이 음수여도 lead_speed가 크면 여전히 실수해가 나오고, 간격이
    너무 좁아 근호 안이 음수가 되면 0(정지). dist가 None이면 무제한."""
    if dist is None:
        return math.inf
    lv = max(lead_speed, 0.0)
    inside = lv * lv + 2.0 * max_decel * (dist - stop_margin)
    if inside <= 0.0:
        return 0.0
    return math.sqrt(inside)


def rule_based_control(
    route_wps: list[tuple[float, float]],
    ego: dict[str, float],
    received: dict[str, list[dict]],
    speed_pi: SpeedPIController,
    dt: float,
    lane_map: LaneMap | None = None,
    cruise_speed: float = CRUISE_SPEED_MPS,
    light_stop_margin_m: float = STOP_LIGHT_DIST_M,
    ego_half_len_m: float = 2.4,
    vehicle_gap_m: float = VEHICLE_GAP_M,
) -> dict[str, float]:
    """모델 없이: route 추종 조향 + 신호등/선행 장애물 규칙 기반 속도로 throttle/steer/brake 계산."""
    x, y, yaw, speed = ego["x"], ego["y"], ego["yaw"], ego["speed"]

    route_ego = _route_ego_ahead(route_wps, x, y, yaw)
    steer = _pure_pursuit_steer(route_ego, speed)

    # 신호등: 내 위치와 헤딩에 가장 가깝게 매칭되는 차선의 신호등 + route가 지나가는 정지선을
    # 함께 본다(둘 중 더 가까운 쪽) — 아래 forced-stop과 맞물려 "정지선 몇 m 전이면 반드시
    # 정지"를 보장한다.
    light_dist = _traffic_light_stop_dist(lane_map, received["infrastructure"], x, y, yaw, route_ego=route_ego)
    # 앞차: 같은 차선(내 경로 arc 거리) 선행차 + 상대 속도로 예측한 교차/합류차를 함께 감지.
    vehicle_dist, lead = _vehicle_hazard(
        received["vehicles"], x, y, yaw, route_ego, LANE_HALF_WIDTH_M,
    )
    ped_dist, _ = _corridor_ahead(received["pedestrians"], x, y, yaw, PED_LANE_HALF_WIDTH_M, route_ego=route_ego)

    # 선행 차량 정지 여유: 내 앞 범퍼 ↔ 앞차 뒤 범퍼 간격이 vehicle_gap_m가 되도록.
    # vehicle_dist는 중심-중심 전방거리라 여유 = 내 반길이 + 간격 + 앞차 반길이.
    # 앞차 반길이는 UWB 페이로드에 실린 실제 치수(depth/2)를 쓰고 — 차량마다 크기가 다르므로 —
    # 치수가 없으면 내 차량과 동급으로 가정한다.
    lead_half_len_m = (lead["depth"] / 2.0) if lead and lead.get("depth") else ego_half_len_m
    vehicle_stop_margin_m = ego_half_len_m + vehicle_gap_m + lead_half_len_m

    # 앞차의 진행방향(ego 헤딩) 속도 성분 — vel_x/vel_y는 월드프레임이라 ego 전방 단위벡터
    # (cos yaw, sin yaw)에 투영한다. 이걸 ACC 목표속도(sqrt(v_lead^2+2ad))에 넣어, 정지한
    # 앞차엔 완전 정지, 움직이는 앞차엔 그 속도에 수렴하도록 한다.
    lead_speed_fwd = 0.0
    if lead:
        yaw_rad = math.radians(yaw)
        lead_speed_fwd = (
            float(lead.get("vel_x") or 0.0) * math.cos(yaw_rad)
            + float(lead.get("vel_y") or 0.0) * math.sin(yaw_rad)
        )

    # 각 요인이 허용하는 최고 속도 — 이 중 최소가 목표 속도를 결정하고, 그 최소를 만든
    # 요인이 곧 (감속/정지) 이유다. dist는 정지 이유 로그에 거리(m)를 같이 찍기 위해 보관.
    limits = {
        "traffic_light": (
            _speed_limit_for_stop(light_dist, light_stop_margin_m, SPEED_LIMIT_DECEL_MSS), light_dist,
        ),
        "vehicle": (
            _speed_limit_for_stop(
                vehicle_dist, vehicle_stop_margin_m, SPEED_LIMIT_DECEL_MSS, lead_speed=lead_speed_fwd,
            ),
            vehicle_dist,
        ),
        "pedestrian": (_speed_limit_for_stop(ped_dist, STOP_PED_DIST_M, SPEED_LIMIT_DECEL_MSS), ped_dist),
    }
    target_speed = min(cruise_speed, *(v for v, _ in limits.values()))
    speed_ctrl = speed_pi.step(target_speed, speed, dt)
    throttle, brake = speed_ctrl["throttle"], speed_ctrl["brake"]

    # 목표 속도가 정지 수준으로 눌렸으면, 가장 강하게 제한한(최소 허용속도) 요인이 정지 이유.
    stop_reason: tuple[str, float] | None = None
    if target_speed <= STOP_REPORT_SPEED_MPS:
        cause = min(limits, key=lambda k: limits[k][0])
        stop_reason = (cause, limits[cause][1])

    # 정지선 여유 거리(light_stop_margin_m) 안까지 들어온 상태에서 여전히 정지 대상 신호라면,
    # PI 제어기의 잔차·지연에 기대지 않고 즉시 완전 정지를 강제한다("정지선 몇 m 전이면 반드시
    # 정지"). speed_limit_for_stop이 이미 그 구간에서 target_speed=0을 만들지만, 노이즈로 인한
    # 순간적인 light_dist 미검출(다음 tick 재검출)까지 대비해 여기서 한 번 더 못박는다.
    if light_dist is not None and light_dist <= light_stop_margin_m:
        throttle, brake = 0.0, 1.0
        stop_reason = ("traffic_light", light_dist)

    return {
        "throttle": throttle, "steer": steer, "brake": brake,
        "stop_reason": stop_reason,
    }


def run_live(
    host: str = "127.0.0.1",
    port: int = 2000,
    hz: float = 60.0,
    uwb_range: float = 150.0,
    cruise_speed: float = CRUISE_SPEED_MPS,
    route_dest_min_dist: float = 30.0,
    route_min_length: float = 100.0,
    route_resync_dist_m: float = ROUTE_RESYNC_DIST_M,
    initial_speed: float = 2.0,
    show_view: bool = True,
    view_size: tuple[int, int] = (960, 540),
    live_map: bool = True,
    live_map_output: str | Path = "test/live_map.png",
    live_map_every: int | None = None,
) -> None:
    """실행 중인 Carla 서버에 접속해 규칙 기반 로직으로 hero 차량을 실시간 주행."""
    sys.path.insert(0, str(ROOT / "sim_setup" / "sensor_setting"))
    import carla  # noqa: E402  (Carla 서버 머신에만 설치되어 있음)
    import numpy as np  # noqa: E402
    import pygame  # noqa: E402

    from agent.vehicle import _setup_vehicle  # noqa: E402

    from carla_hud import HUD, CollisionSensor, LaneInvasionSensor  # noqa: E402
    from live_map import LiveMapPlotter  # noqa: E402

    build_route_planner = _load_route_planner(sampling_resolution=ROUTE_SAMPLING_RESOLUTION_M)

    lane_map = get_lane_map(MAP_JSON) if MAP_JSON.exists() else None
    if lane_map is None:
        print(
            f"[{_ts()}] [rule_based_test] {MAP_JSON} 없음 → 신호등 정지선 매칭 불가, 신호등 규칙 비활성화 "
            "(sim_setup/map_downloader.py 먼저 실행)"
        )

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()

    orig_settings = world.get_settings()
    world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=1.0 / hz))

    hero, spawned = _find_or_spawn_hero(world, random_spawn=True)
    print(f"[{_ts()}] [rule_based_test] hero id={hero.id} type={hero.type_id} (spawned={spawned})")

    # 정지 거리 여유는 차량 외형 기준으로 잡는다. UWB/IMU는 차량 중심(x,y=0 오프셋)에 mount돼
    # 있어 light_dist/vehicle_dist는 "내 차량 중심 → 대상"까지의 전방 거리다. half_len_m은
    # 중심→앞 범퍼 거리(extent.x).
    #  - 신호등: 앞 범퍼가 정지선 1.5m 전에 서도록 → 중심은 정지선 (half_len + 1.5)m 전에 정지
    #  - 선행 차량: 내 앞 범퍼 ↔ 앞차 뒤 범퍼 간격 VEHICLE_GAP_M. 앞차 반길이는 차량마다 다르므로
    #    여유값을 고정하지 않고 rule_based_control이 매 tick 감지된 앞차의 실제 depth로 계산한다.
    half_len_m = hero.bounding_box.extent.x
    light_stop_margin = half_len_m + 1.5
    print(
        f"[{_ts()}] [rule_based_test] 정지 여유: 신호등 앞범퍼 정지선 1.5m전(중심기준 {light_stop_margin:.1f}m), "
        f"차량 범퍼간 {VEHICLE_GAP_M:.0f}m(앞차 실제 크기 반영, 내 반길이 {half_len_m:.2f}m)"
    )

    if initial_speed > 0:
        fwd = hero.get_transform().get_forward_vector()
        hero.set_target_velocity(carla.Vector3D(x=fwd.x * initial_speed, y=fwd.y * initial_speed, z=0.0))

    grp = build_route_planner(world)
    route_wps: list[tuple[float, float]] = []
    route_dest = None
    off_route_ticks = 0  # est_pos가 route에서 벗어난 상태가 몇 tick 연속인지(노이즈성 재생성 방지)
    # 커밋된 route는 한 tick 노이즈로 버리지 않고, ~0.4초(hz*0.4 tick) 연속으로 이탈일 때만 재생성.
    resync_persist_ticks = max(1, round(hz * 0.4))
    last_stop_cause: str | None = None  # 같은 정지 이유를 매 tick 반복 출력하지 않도록(상황당 1회)

    map_plotter = None
    if live_map:
        if MAP_JSON.exists():
            map_plotter = LiveMapPlotter(
                MAP_JSON, live_map_output, update_every=live_map_every or max(1, round(hz)),
            )
            print(f"[{_ts()}] [rule_based_test] live map → {Path(live_map_output).resolve()} (계속 갱신됨)")
        else:
            print(f"[{_ts()}] [rule_based_test] live map 비활성화 — {MAP_JSON} 없음 (sim_setup/map_downloader.py 먼저 실행)")

    bp_lib = world.get_blueprint_library()
    mount = carla.Transform(carla.Location(z=1.5))
    uwb_bp = bp_lib.find("sensor.other.uwb")
    uwb_bp.set_attribute("max_range", str(uwb_range))
    imu = world.spawn_actor(bp_lib.find("sensor.other.imu"), mount, attach_to=hero)
    uwb = world.spawn_actor(uwb_bp, mount, attach_to=hero)
    _, state_ref, lock = _setup_vehicle(hero, uwb, imu, send=False)

    # pygame.init()과 clock은 뷰 표시(HUD)를 위해 필요하다. 다만 룰베이스에서는 프레임 레이트를
    # 아예 관리하지 않는다 — synchronous 모드라 world.tick()이 서버 속도대로 배속으로 흐르는데,
    # 메인 루프에서 clock.tick(hz)로 실시간 페이싱을 하면 서버가 못 따라올 때 프레임이 드랍돼
    # 끊긴다. 배속을 감수하고 페이싱 없이 매끄럽게 돌린다.
    pygame.init()
    clock = pygame.time.Clock()

    view = None
    if show_view:
        ext = hero.bounding_box.extent
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(view_size[0]))
        cam_bp.set_attribute("image_size_y", str(view_size[1]))
        cam_transform = carla.Transform(
            carla.Location(x=-2.0 * ext.x, z=2.0 * ext.z + 1.0), carla.Rotation(pitch=8.0),
        )
        camera = world.spawn_actor(
            cam_bp, cam_transform, attach_to=hero,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )
        pygame.font.init()
        display = pygame.display.set_mode(view_size)
        pygame.display.set_caption("rule_based_test — hero 3인칭 뷰 (manual_control.py 동일 HUD)")
        hud = HUD(*view_size)
        world.on_tick(hud.on_world_tick)
        collision_sensor = CollisionSensor(hero, hud, carla)
        lane_invasion_sensor = LaneInvasionSensor(hero, hud, carla)
        hud.notification(f"{'Spawned' if spawned else 'Found'} hero: {hero.type_id}")
        view = {
            "camera": camera,
            "display": display,
            "surface": None,
            "hud": hud,
            "collision": collision_sensor,
            "lane_invasion": lane_invasion_sensor,
            "clock": clock,
        }

        def _on_camera_image(image, _view=view):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
            _view["surface"] = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        camera.listen(_on_camera_image)

    stop_ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    stopped = {"flag": False}

    def cleanup(*_args) -> None:
        if stopped["flag"]:
            return
        stopped["flag"] = True
        print(f"\n[{_ts()}] [rule_based_test] 정리 중...")
        for s in (imu, uwb):
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass
        if view is not None:
            try:
                view["collision"].destroy()
                view["lane_invasion"].destroy()
                view["camera"].stop()
                view["camera"].destroy()
            except Exception:
                pass
            pygame.quit()
        if spawned:
            try:
                hero.destroy()
            except Exception:
                pass
        if map_plotter is not None:
            map_plotter.close()
        world.apply_settings(orig_settings)
        print(f"[{_ts()}] [rule_based_test] Done.")

    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(0)))

    map_name = world.get_map().name.split("/")[-1]

    def _render_view(model_status: str, control: dict[str, float] | None) -> None:
        if view is None:
            return
        with lock:
            loc = state_ref["est_pos"]
            yaw_deg = state_ref["imu"]["yaw"]
            yaw_rate_deg = math.degrees(state_ref["imu"]["yaw_rate"])
            spd = state_ref["wheel"]["speed"]
            accel = (
                state_ref["wheel"]["acc_x"], state_ref["wheel"]["acc_y"], state_ref["wheel"]["acc_z"],
            )
        location = (loc[0] or 0.0, loc[1] or 0.0, loc[2] or 0.0)
        route_left = _route_length(route_wps, location[0], location[1]) if route_wps else None
        view["hud"].tick(
            view["clock"], hero, map_name, location, spd, yaw_deg, accel, yaw_rate_deg,
            control, view["collision"], route_left, model_status,
        )
        if view["surface"] is not None:
            view["display"].blit(view["surface"], (0, 0))
        view["hud"].render(view["display"])
        pygame.display.flip()

    # 순항 속도 추종 PI 제어기 — P만 쓰면 목표보다 낮은 속도로 수렴(정상상태 오차)하므로
    # 적분항으로 잔여 오차를 지운다. 감속도 상한은 정지거리 역산에 쓰는 값과 동일하게 맞춘다.
    speed_pi = SpeedPIController(kp=0.6, ki=0.5, max_accel_mss=MAX_DECEL_MSS)

    print(f"[{_ts()}] [rule_based_test] {hz} Hz 동기 모드 — 규칙 기반 주행 시작 (순항 속도 {cruise_speed} m/s)")
    try:
        while True:
            world.tick()
            # 룰베이스에서는 프레임 레이트를 아예 관리하지 않는다 — clock.tick(hz)로 페이싱하면
            # 서버가 못 따라올 때 프레임이 드랍돼 끊기고, 인자 없는 tick()조차 굳이 프레임을
            # 건드릴 이유가 없다. world.tick()이 서버 속도대로(배속) 매끄럽게 돌게 둔다.
            if view is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        cleanup()
                        sys.exit(0)
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_ESCAPE:
                            cleanup()
                            sys.exit(0)
                        elif event.key == pygame.K_F1:
                            view["hud"].toggle_info()
                        elif event.key in (pygame.K_h, pygame.K_SLASH):
                            view["hud"].help.toggle()
            with lock:
                x, y = state_ref["est_pos"][0], state_ref["est_pos"][1]
                yaw = state_ref["imu"]["yaw"]
                speed = state_ref["wheel"]["speed"]
                raw_received = {k: dict(p) for k, p in state_ref["received"].items()}

            if x is None:
                print(f"[{_ts()}] [rule_based_test] 위치 추정 안됨 (UWB anchor 부족) → stop")
                hero.apply_control(stop_ctrl)
                _render_view("no UWB fix", None)
                continue

            # route는 "하나에 커밋 후, 남은 길이가 입력 route 길이(route_min_length) 이하일 때만
            # 목적지 연장" 방식으로 유지한다. 중간에 route를 다시 뽑지 않으므로 갑자기 안 바뀐다.
            prev_len = len(route_wps)
            route_wps, route_dest, off_route_ticks, route_events = _maintain_route(
                world, grp, carla, route_wps, route_dest, x, y, yaw, off_route_ticks,
                route_dest_min_dist=route_dest_min_dist,
                input_route_length_m=route_min_length,
                resync_dist_m=route_resync_dist_m,
                resync_persist_ticks=resync_persist_ticks,
            )
            for msg in route_events:
                print(f"[{_ts()}] [rule_based_test] {msg}")
            if view is not None and route_events and len(route_wps) != prev_len:
                view["hud"].notification(f"Route: {len(route_wps)} waypoints")

            ego = {"x": x, "y": y, "yaw": yaw, "speed": speed}
            received = _sort_received(raw_received, x, y)
            route_ahead = route_wps[:ROUTE_LOOKAHEAD_POINTS]
            ctrl = rule_based_control(
                route_ahead, ego, received, speed_pi, 1.0 / hz,
                lane_map=lane_map, cruise_speed=cruise_speed,
                light_stop_margin_m=light_stop_margin, ego_half_len_m=half_len_m, vehicle_gap_m=VEHICLE_GAP_M,
            )

            if map_plotter is not None:
                map_plotter.update((x, y), route_ahead, ego_yaw_deg=yaw)

            # 정지 이유는 그 상황이 시작될 때(또는 이유가 바뀔 때) 딱 1회만 출력 — 멈춰 있는
            # 동안 매 tick 같은 줄을 쏟아내지 않도록 직전 이유(last_stop_cause)와 비교한다.
            # 정지가 풀려 다시 출발할 때도(cause: 뭔가 → None) 한 번 출력해 로그만 보고도
            # "언제부터 언제까지 멈춰 있었는지"를 알 수 있게 한다.
            stop_reason = ctrl.get("stop_reason")
            cause = stop_reason[0] if stop_reason else None
            if cause != last_stop_cause:
                label_map = {"traffic_light": "신호등(red/yellow)", "vehicle": "선행 차량", "pedestrian": "보행자"}
                if cause is not None:
                    print(f"[{_ts()}] [rule_based_test] 정지: {label_map[cause]} (거리 {stop_reason[1]:.1f}m)")
                elif last_stop_cause is not None:
                    print(f"[{_ts()}] [rule_based_test] 정지 해제: {label_map[last_stop_cause]} → 재출발")
                last_stop_cause = cause

            # print(
            #     f"[{_ts()}] [rule_based_test] ctrl t/s/b=({ctrl['throttle']:.3f},{ctrl['steer']:.3f},{ctrl['brake']:.3f})"
            # )
            hero.apply_control(carla.VehicleControl(
                throttle=float(ctrl["throttle"]),
                steer=float(ctrl["steer"]),
                brake=float(ctrl["brake"]),
            ))
            _render_view("rule-based driving", ctrl)
    finally:
        cleanup()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carla 규칙 기반(rule-based) 주행 테스트")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument(
        "--hz", type=float, default=60.0,
        help="동기 모드 tick(및 렌더) 주파수. synchronous 모드에서는 tick당 렌더 1장이라 "
             "hz가 곧 화면 fps 상한이 됨 — 모델과 달리 규칙 기반은 dt에 종속적인 학습 가정이 "
             "없어서 부드러운 화면을 위해 기본값을 높게 잡음",
    )
    p.add_argument("--uwb-range", type=float, default=150.0)
    p.add_argument(
        "--cruise-speed", type=float, default=CRUISE_SPEED_MPS,
        help="전방이 비어 있을 때 유지할 순항 속도(m/s)",
    )
    p.add_argument(
        "--route-dest-min-dist", type=float, default=30.0,
        help="랜덤 목적지 선택 시 현재 위치에서 최소 이 거리(m) 이상 떨어진 spawn point만 후보로 사용",
    )
    p.add_argument(
        "--route-min-length", type=float, default=INPUT_ROUTE_LENGTH_M,
        help="남은 route 길이(m)가 이 값 이하로 줄면 현재 목적지에서 다른 목적지까지 route를 이어 붙임. "
             "기본값은 입력으로 나가는 route_ahead 길이(ROUTE_LOOKAHEAD_POINTS*ROUTE_SAMPLING_RESOLUTION_M)와 동일 — "
             "커밋된 route를 끝까지 따르다 앞부분만 남았을 때만 연장한다",
    )
    p.add_argument(
        "--route-resync-dist-m", type=float, default=ROUTE_RESYNC_DIST_M,
        help="ego(est_pos)와 route 사이 거리가 이 값(m)을 넘는 상태가 여러 tick 연속되면(노이즈성 1회는 무시) "
             "커밋된 route를 버리고 현재 위치 기준으로 새로 만듦",
    )
    p.add_argument(
        "--initial-speed", type=float, default=2.0,
        help="스폰 직후 진행 방향으로 실어줄 초기 속도(m/s). 0이면 끔",
    )
    p.add_argument("--no-view", action="store_true", help="hero 3인칭 뷰 pygame 창을 띄우지 않음")
    p.add_argument("--view-width", type=int, default=960)
    p.add_argument("--view-height", type=int, default=540)
    p.add_argument(
        "--no-live-map", action="store_true",
        help="맵 위에 driven trajectory/route를 그려 PNG로 계속 갱신하는 기능을 끔",
    )
    p.add_argument("--live-map-output", default="test/live_map.png")
    p.add_argument("--live-map-every", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_live(
        host=args.host,
        port=args.port,
        hz=args.hz,
        uwb_range=args.uwb_range,
        cruise_speed=args.cruise_speed,
        route_dest_min_dist=args.route_dest_min_dist,
        route_min_length=args.route_min_length,
        route_resync_dist_m=args.route_resync_dist_m,
        initial_speed=args.initial_speed,
        show_view=not args.no_view,
        view_size=(args.view_width, args.view_height),
        live_map=not args.no_live_map,
        live_map_output=args.live_map_output,
        live_map_every=args.live_map_every,
    )

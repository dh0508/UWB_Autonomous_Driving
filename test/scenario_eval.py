"""시나리오 기반 정량 평가 — "정지 필요 이벤트"의 대응 성공률을 측정한다.

rule_based_test.py와 동일하게 Carla 서버에 접속해 hero를 주행시키되,
주행 자체는 평가하지 않고 **매 tick CARLA 지상진실(ground-truth)을 관측**해 아래 세 종류의
"정지가 필요한 이벤트"가 몇 번 발생했고, 각 이벤트에 ego가 제대로 대응(정지/안전 통과)했는지
실패(충돌·위험 근접·신호 위반)했는지를 세어 다음과 같은 표를 만든다.

    이벤트            발생 횟수  정상 처리  실패  성공률
    정지 차량 대응
    적색신호 접근
    보행자 횡단
    전체 정지 필요 이벤트

판정은 주행 로직이 보는 UWB 인지값이 아니라 **CARLA world 액터의 실제 상태**로 한다
(실주행 안전 결과를 정직하게 재기 위함):
  - ego 실제 pose/속도 : hero.get_transform() / hero.get_velocity()
  - 주변 차량/보행자    : world.get_actors().filter("vehicle.*" / "walker.pedestrian.*")
  - 신호등              : hero.get_traffic_light() / hero.get_traffic_light_state()
  - 충돌                : sensor.other.collision (상대 액터 id까지 기록해 이벤트에 귀속)

── 실행 방법 (test/carla_common.py 준비 절차와 동일, 이 스크립트가 hero 컨트롤러 자리) ──
  1. Carla 서버 실행
  2. sim_setup/scenarios/generate_traffic.py -n 30 -w 8 -s <seed>  (NPC 차량 30 · 보행자 8)
  3. sim_setup/sensor_setting/sensor_attach.py
  4. sim_setup/sensor_setting/agent/infrastructure.py  (+ 원하면 agent/vehicle.py, agent/pedestrian.py)
  5. python test/scenario_eval.py --seed <seed> --duration 600 --results-json test/eval_results.json

시드마다 (2)~(5)를 반복하면 각 세션 결과가 --results-json에 누적되고, 매 실행 끝에
지금까지 누적된 **전체 시드 집계 표**가 출력된다. 집계만 다시 보려면:
    python test/scenario_eval.py --report --results-json test/eval_results.json

주의: NPC/보행자 수·종류·초기 위치는 generate_traffic.py의 seed가 정하고, 이 스크립트의
--seed는 hero 스폰 지점(및 랜덤 목적지)을 재현 가능하게 만드는 데 쓴다. 실제 NPC/보행자
수는 world에서 읽어 리포트 헤더에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
import weakref
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from data.src.control import SpeedPIController  # noqa: E402
from data.src.map_loader import get_lane_map  # noqa: E402
from data.src.paths import MAP_JSON  # noqa: E402

from carla_common import (  # noqa: E402
    ROUTE_RESYNC_DIST_M,
    _find_or_spawn_hero,
    _load_route_planner,
    _sort_received,
)
from rule_based_test import (  # noqa: E402
    CRUISE_SPEED_MPS,
    MAX_DECEL_MSS,
    ROUTE_LOOKAHEAD_POINTS,
    ROUTE_SAMPLING_RESOLUTION_M,
    VEHICLE_GAP_M,
    _maintain_route,
    rule_based_control,
)

# ── 이벤트 판정 임계값 ────────────────────────────────────────────────────────
# 모두 CARLA 지상진실 기준. 코리도(같은 차선/경로) 판정은 ego 헤딩 기준 전방/측방 직선축으로
# 단순화한다 — 반응거리(react)가 짧아(≤25 m) 커브 왜곡이 작다.
EGO_MOVING_MPS = 1.0          # ego가 "주행 중"으로 볼 최소 속도
EGO_STOPPED_MPS = 0.6         # ego가 "정지함"으로 볼 최대 속도 (정지 대응 성공 판정)
OTHER_STOPPED_MPS = 0.6       # 상대 차량이 "정지 차량"으로 볼 최대 속도

VEH_DETECT_M = 25.0           # 정지 차량을 이벤트 후보로 감지할 최대 전방거리(중심-중심)
VEH_REACT_M = 25.0            # 이 거리 안으로 들어오면 "실제로 정지가 필요한 상황"으로 카운트
                               # (VEH_DETECT_M과 동일 — 정지 여유가 늘어나 ego가 감지 범위 안
                               #  어디서든 이미 서행/정지를 시작할 수 있어, react_m을 좁게 두면
                               #  "멀리서 일찍 잘 정지한" 성공 사례가 counted 문턱(ego_speed>
                               #  EGO_MOVING_MPS)을 못 넘겨 이벤트로 안 잡히고 누락된다.)
VEH_CORRIDOR_HALF_M = 1.6     # 같은 차선/경로로 볼 측방 반폭(상대 폭/2는 별도로 더함)

PED_DETECT_M = 20.0
PED_REACT_M = 15.0
PED_CORRIDOR_HALF_M = 2.0

# hero.get_traffic_light()는 CARLA map에 그려진 trigger_volume 안에 있을 때만 값을 준다.
# ego가 그보다 더 일찍 정지하면 get_traffic_light()가 None만 반환해 이벤트가 안 열리고
# 그 정지가 누락된다. trigger_volume 밖으로 이 마진(m)만큼 더 앞까지
# get_traffic_lights_from_waypoint로 다음 신호등을 직접 찾아 보완한다.
RED_LIGHT_DETECT_MARGIN_M = 10.0

# 정지선의 진행방향(_stop_line_for의 fx, fy)이 ego 헤딩과 이 코사인값 이상으로 정렬돼야
# "내 신호"로 인정한다 — 교차로에서 hero.get_traffic_light()/get_traffic_lights_from_waypoint가
# 거리만 보고 좌우회전·교차 도로용 신호(내 진행방향이 아닌 신호)를 집어오는 오판을 막기 위함.
# cos(60°)=0.5: 대략 ±60° 이내로 정렬된 신호만 인정.
RED_LIGHT_HEADING_COS_MIN = 0.5

# 범퍼 표면 간격이 이 값(m) 아래로 좁혀지면 충돌이 없어도 "위험 근접(대응 실패)"으로 본다.
FAIL_SURFACE_GAP_M = 0.4

# 코리도를 벗어난 뒤 이 시간(초) 동안 재진입이 없으면 이벤트 종료로 확정(깜빡임 방지).
EPISODE_GRACE_S = 1.0


def _ts() -> str:
    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"


def _ego_frame(dx: float, dy: float, yaw_deg: float) -> tuple[float, float]:
    """world 상대변위(dx, dy)를 ego 프레임(forward=x축, lateral=우측 y축)으로 회전."""
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    forward = dx * cos_y + dy * sin_y
    lateral = -dx * sin_y + dy * cos_y
    return forward, lateral


class EvalCollisionSensor:
    """hero 충돌을 상대 액터 id와 함께 기록 — 어떤 이벤트(차량/보행자)의 실패인지 귀속하기 위함.

    carla_hud.CollisionSensor는 intensity만 남기고 상대 액터를 버리므로 평가용으로 새로 둔다.
    콜백은 CARLA 리스너 스레드에서 호출되지만, 기록하는 값(액터 id → 마지막 충돌 시각)은
    단순 dict 대입이라 GIL 하에서 안전하다."""

    def __init__(self, hero, carla_module) -> None:
        self.collided_at: dict[int, float] = {}   # other_actor_id -> 최근 충돌 시각(sim s)
        self._now = 0.0
        world = hero.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla_module.Transform(), attach_to=hero)
        weak_self = weakref.ref(self)

        def _on(event, _wref=weak_self):
            self_ = _wref()
            if self_ is None:
                return
            self_.collided_at[event.other_actor.id] = self_._now

        self.sensor.listen(_on)

    def set_time(self, now_s: float) -> None:
        self._now = now_s

    def hit_recently(self, actor_id: int, now_s: float, window_s: float = 2.0) -> bool:
        t = self.collided_at.get(actor_id)
        return t is not None and (now_s - t) <= window_s

    def destroy(self) -> None:
        try:
            if self.sensor is not None and self.sensor.is_alive:
                self.sensor.stop()
                self.sensor.destroy()
        except Exception:
            pass


class _Tally:
    """한 이벤트 종류의 누적 집계 (발생/성공/실패)."""

    def __init__(self) -> None:
        self.success = 0
        self.fail = 0

    @property
    def total(self) -> int:
        return self.success + self.fail

    def add(self, ok: bool) -> None:
        if ok:
            self.success += 1
        else:
            self.fail += 1

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "success": self.success, "fail": self.fail}


class ScenarioEvaluator:
    """매 tick CARLA 지상진실을 관측해 정지 필요 이벤트를 에피소드 단위로 추적·판정한다."""

    def __init__(self, hero, collision: EvalCollisionSensor, grace_s: float = EPISODE_GRACE_S) -> None:
        self.hero = hero
        self.hero_id = hero.id
        self.collision = collision
        self.grace_s = grace_s
        self.ego_half_len = float(hero.bounding_box.extent.x)

        self.veh = _Tally()
        self.ped = _Tally()
        self.red = _Tally()

        # 열려 있는 에피소드들. 차량/보행자는 액터 id로, 신호등은 tl id로 키를 잡아
        # tick 간 같은 대상을 하나의 에피소드로 잇는다.
        self._veh_eps: dict[int, dict[str, Any]] = {}
        self._ped_eps: dict[int, dict[str, Any]] = {}
        self._red_eps: dict[int, dict[str, Any]] = {}

        self.distance_m = 0.0
        self._last_xy: tuple[float, float] | None = None

    # ── 공개 API ──────────────────────────────────────────────────────────
    def tick(self, now_s: float, world) -> None:
        self.collision.set_time(now_s)

        tf = self.hero.get_transform()
        ex, ey = tf.location.x, tf.location.y
        yaw = tf.rotation.yaw
        vel = self.hero.get_velocity()
        ego_speed = math.hypot(vel.x, vel.y)

        if self._last_xy is not None:
            self.distance_m += math.hypot(ex - self._last_xy[0], ey - self._last_xy[1])
        self._last_xy = (ex, ey)

        actors = world.get_actors()
        self._update_actor_events(
            now_s, ex, ey, yaw, ego_speed,
            actors.filter("vehicle.*"), self._veh_eps, self.veh,
            detect_m=VEH_DETECT_M, react_m=VEH_REACT_M, corridor_half_m=VEH_CORRIDOR_HALF_M,
            require_stopped=True,
        )
        self._update_actor_events(
            now_s, ex, ey, yaw, ego_speed,
            actors.filter("walker.pedestrian.*"), self._ped_eps, self.ped,
            detect_m=PED_DETECT_M, react_m=PED_REACT_M, corridor_half_m=PED_CORRIDOR_HALF_M,
            require_stopped=False,
        )
        self._update_red_light_events(now_s, ego_speed, ex, ey, yaw, world)

    def finalize(self) -> None:
        """세션 종료 시 아직 열려 있는(카운트된) 에피소드를 현재 상태로 확정한다."""
        for eps, tally in ((self._veh_eps, self.veh), (self._ped_eps, self.ped)):
            for ep in eps.values():
                if ep["counted"]:
                    tally.add(not ep["failed"])
            eps.clear()
        for ep in self._red_eps.values():
            # 아직 정지 못 했고 적색이면 실패, 정지했으면 성공으로 마감.
            self.red.add(ep["stopped"])
        self._red_eps.clear()

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            "stopped_vehicle": self.veh.as_dict(),
            "red_light": self.red.as_dict(),
            "pedestrian": self.ped.as_dict(),
        }

    def active_counts(self) -> dict[str, int]:
        """아직 마감되지 않은(진행 중) 이벤트 수 — 실시간 대시보드 표시용."""
        return {
            "stopped_vehicle": sum(1 for ep in self._veh_eps.values() if ep["counted"]),
            "red_light": len(self._red_eps),
            "pedestrian": sum(1 for ep in self._ped_eps.values() if ep["counted"]),
        }

    # ── 내부: 차량/보행자 에피소드 ────────────────────────────────────────
    def _update_actor_events(
        self, now_s: float, ex: float, ey: float, yaw: float, ego_speed: float,
        candidates, eps: dict[int, dict[str, Any]], tally: _Tally,
        *, detect_m: float, react_m: float, corridor_half_m: float, require_stopped: bool,
    ) -> None:
        for actor in candidates:
            if actor.id == self.hero_id:
                continue
            loc = actor.get_location()
            forward, lateral = _ego_frame(loc.x - ex, loc.y - ey, yaw)
            if forward <= 0.0 or forward > detect_m:
                continue

            ext = actor.bounding_box.extent
            half_w = float(ext.y)
            half_len = float(ext.x)
            if abs(lateral) - half_w > corridor_half_m:
                continue  # 같은 차선/경로 코리도 밖

            ep = eps.get(actor.id)
            if require_stopped:
                av = actor.get_velocity()
                is_stopped = math.hypot(av.x, av.y) <= OTHER_STOPPED_MPS
                # 처음 보는(아직 에피소드 없는) 차가 정지 상태가 아니면 "정지 차량 대응" 이벤트
                # 대상이 아니다. 이미 에피소드가 열려 있는 차라면, 정지·재출발이 섞인 서행
                # 구간에서 잠깐 속도가 올라도(신호 대기열 등) last_seen을 계속 갱신해 같은
                # 만남을 하나의 에피소드로 유지한다 — 안 그러면 grace_s(1.0초) 넘게 서행하는
                # 순간 에피소드가 조기 마감·집계되고, 다시 멈추면 같은 차가 새 에피소드로 또
                # 카운트돼 "한 차량의 정지가 여러 번 성공으로 오르는" 중복 집계가 생긴다.
                if not is_stopped and ep is None:
                    continue

            surface_gap = forward - self.ego_half_len - half_len

            if ep is None:
                ep = {"counted": False, "failed": False, "last_seen": now_s}
                eps[actor.id] = ep
            ep["last_seen"] = now_s

            # 반응이 실제로 필요한 상황(반응거리 안 + ego 주행 중)에서만 이벤트로 카운트.
            if forward <= react_m and (ego_speed > EGO_MOVING_MPS or ep["counted"]):
                ep["counted"] = True

            # 실패: 물리 충돌(귀속) 또는 범퍼 표면 임계거리 침범.
            if self.collision.hit_recently(actor.id, now_s) or surface_gap < FAIL_SURFACE_GAP_M:
                ep["failed"] = True

        # 코리도를 벗어난 지 grace_s 초과한 에피소드는 종료 확정.
        for actor_id in [k for k, ep in eps.items() if now_s - ep["last_seen"] > self.grace_s]:
            ep = eps.pop(actor_id)
            if ep["counted"]:
                tally.add(not ep["failed"])

    # ── 내부: 적색신호 에피소드 ───────────────────────────────────────────
    def _stop_line_for(self, tl, ex: float, ey: float) -> tuple[float, float, float, float] | None:
        """tl에 연결된 정지선(들) 중 ego와 가장 가까운 것의 (x, y, 전방벡터x, 전방벡터y).

        hero.get_traffic_light()는 물리적 정지선보다 앞으로 넓게 퍼진 trigger_volume 안에
        있는 동안 계속 그 신호등을 반환하므로, 속도만으로 "정지 성공"을 판정하면 이미 정지선을
        넘어선 뒤 멈춘 경우(신호 위반)도 성공으로 오판한다. get_stop_waypoints()로 실제 정지선
        좌표(및 차선 진행방향)를 얻어 ego가 그 선을 넘었는지 함께 확인한다."""
        try:
            wps = tl.get_stop_waypoints()
        except Exception:
            wps = None
        if not wps:
            return None
        best = None
        best_d_sq = math.inf
        for wp in wps:
            loc = wp.transform.location
            d_sq = (loc.x - ex) ** 2 + (loc.y - ey) ** 2
            if d_sq < best_d_sq:
                best_d_sq = d_sq
                fwd = wp.transform.get_forward_vector()
                best = (loc.x, loc.y, fwd.x, fwd.y)
        return best

    @staticmethod
    def _past_stop_line(stop_line: tuple[float, float, float, float] | None, ex: float, ey: float) -> bool:
        if stop_line is None:
            return False
        sx, sy, fx, fy = stop_line
        return (ex - sx) * fx + (ey - sy) * fy > 0.0

    @staticmethod
    def _aligned_with_ego(
        stop_line: tuple[float, float, float, float] | None, yaw_deg: float,
    ) -> bool:
        """정지선 진행방향이 ego 헤딩과 충분히 정렬됐는지 — 교차로에서 다른 방향(좌우회전·
        교차 도로) 신호를 "내 신호"로 오판하는 것을 막기 위한 체크."""
        if stop_line is None:
            return False
        _, _, fx, fy = stop_line
        yaw = math.radians(yaw_deg)
        ego_fx, ego_fy = math.cos(yaw), math.sin(yaw)
        return (fx * ego_fx + fy * ego_fy) >= RED_LIGHT_HEADING_COS_MIN

    def _find_upcoming_red_light(self, world, ex: float, ey: float, yaw: float):
        """trigger_volume 밖이라 hero.get_traffic_light()가 None일 때, 그보다
        RED_LIGHT_DETECT_MARGIN_M만큼 더 앞까지 다음 신호등을 직접 찾아본다.
        ego 헤딩과 정렬된(=내 진행방향) 정지선을 가진 후보만 고려한다."""
        try:
            wp = world.get_map().get_waypoint(self.hero.get_location())
            candidates = world.get_traffic_lights_from_waypoint(wp, RED_LIGHT_DETECT_MARGIN_M)
        except Exception:
            candidates = []
        best, best_d_sq = None, math.inf
        for c in candidates:
            stop_line = self._stop_line_for(c, ex, ey)
            if stop_line is None or not self._aligned_with_ego(stop_line, yaw):
                continue
            d_sq = (stop_line[0] - ex) ** 2 + (stop_line[1] - ey) ** 2
            if d_sq < best_d_sq:
                best_d_sq, best = d_sq, c
        return best

    def _update_red_light_events(
        self, now_s: float, ego_speed: float, ex: float, ey: float, yaw: float, world,
    ) -> None:
        tl = None
        try:
            tl = self.hero.get_traffic_light()
        except Exception:
            tl = None
        if tl is not None and not self._aligned_with_ego(self._stop_line_for(tl, ex, ey), yaw):
            # 교차로 등에서 trigger_volume은 겹쳐 있지만 내 진행방향 신호가 아닌 경우 배제.
            tl = None
        if tl is None:
            tl = self._find_upcoming_red_light(world, ex, ey, yaw)
        cur_id = tl.id if tl is not None else None

        if tl is not None:
            state = str(tl.state)
            ep = self._red_eps.get(cur_id)
            if ep is None:
                # 정지 의무가 있는 적색만 이벤트로 카운트한다 — 황색은 정지선을 넘어도
                # 신호 위반이 아니므로 실패 대상에서 제외.
                if state == "Red":
                    stop_line = self._stop_line_for(tl, ex, ey)
                    # 적색으로 바뀌기 전에 이미 정지선을 통과했다면(=합법적으로 진입 완료) 위반이
                    # 아니므로 이벤트 자체를 열지 않는다.
                    if not self._past_stop_line(stop_line, ex, ey):
                        ep = {"stopped": False, "state": state, "stop_line": stop_line}
                        self._red_eps[cur_id] = ep
            if ep is not None:
                ep["state"] = state
                if ego_speed < EGO_STOPPED_MPS and not self._past_stop_line(ep["stop_line"], ex, ey):
                    ep["stopped"] = True
                if state == "Green":
                    # 적색으로 접근했다가 녹색 전환 후 통과 → 정상 처리.
                    self.red.add(True)
                    self._red_eps.pop(cur_id, None)

        # 영향 신호가 바뀌었는데(=그 신호를 통과·이탈) 마감 안 된 에피소드 판정.
        for tl_id in [k for k in self._red_eps if k != cur_id]:
            ep = self._red_eps.pop(tl_id)
            # 정지한 적 있으면 정상 처리, 정지 없이 적색을 지나쳤으면 신호 위반(실패).
            self.red.add(ep["stopped"])


# ── 집계/리포트 ──────────────────────────────────────────────────────────────
EVENT_LABELS = [
    ("stopped_vehicle", "정지 차량 대응"),
    ("red_light", "적색신호 접근"),
    ("pedestrian", "보행자 횡단"),
]


def _blank_counts() -> dict[str, dict[str, int]]:
    return {k: {"total": 0, "success": 0, "fail": 0} for k, _ in EVENT_LABELS}


def _disp_width(s: str) -> int:
    """모노스페이스 터미널 표시 폭 — 한글 등 전각 문자는 2칸으로 센다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    """표시 폭 기준 좌측 정렬 패딩(전각 문자 폭 보정)."""
    return s + " " * max(0, width - _disp_width(s))


def _rpad(s: str, width: int) -> str:
    """표시 폭 기준 우측 정렬 패딩(전각 문자 폭 보정)."""
    return " " * max(0, width - _disp_width(s)) + s


def aggregate(runs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    """저장된 run 리스트를 하나의 집계 counts + 메타 합계로 합친다."""
    counts = _blank_counts()
    meta = {"seeds": 0, "drive_min": 0.0, "distance_km": 0.0, "npc_vehicles": 0, "pedestrians": 0}
    for run in runs:
        for key, _ in EVENT_LABELS:
            c = run.get("counts", {}).get(key, {})
            for f in ("total", "success", "fail"):
                counts[key][f] += int(c.get(f, 0))
        m = run.get("meta", {})
        meta["seeds"] += 1
        meta["drive_min"] += float(m.get("drive_min", 0.0))
        meta["distance_km"] += float(m.get("distance_km", 0.0))
        meta["npc_vehicles"] += int(m.get("npc_vehicles", 0))
        meta["pedestrians"] += int(m.get("pedestrians", 0))
    return counts, meta


def format_table(counts: dict[str, dict[str, int]], meta: dict[str, float]) -> str:
    seeds = int(meta.get("seeds", 0)) or 1

    def rate(c: dict[str, int]) -> str:
        return f"{100.0 * c['success'] / c['total']:.1f}%" if c["total"] else "  -  "

    total = {"total": 0, "success": 0, "fail": 0}
    for key, _ in EVENT_LABELS:
        for f in total:
            total[f] += counts[key][f]

    lines: list[str] = []
    lines.append("=" * 56)
    lines.append(" 시나리오 기반 정지-필요 이벤트 평가")
    lines.append("=" * 56)
    lines.append(f"  랜덤 시드          : {seeds}개")
    lines.append(f"  NPC 차량(누적)     : {int(meta.get('npc_vehicles', 0))}대"
                 f"  (시드당 평균 {meta.get('npc_vehicles', 0) / seeds:.0f}대)")
    lines.append(f"  보행자(누적)       : {int(meta.get('pedestrians', 0))}명"
                 f"  (시드당 평균 {meta.get('pedestrians', 0) / seeds:.0f}명)")
    lines.append(f"  총 주행시간        : {meta.get('drive_min', 0.0):.1f}분"
                 f"  (회당 평균 {meta.get('drive_min', 0.0) / seeds:.1f}분)")
    lines.append(f"  총 주행거리        : {meta.get('distance_km', 0.0):.2f} km")
    lines.append("-" * 56)
    lines.append(f"  {_pad('이벤트', 18)}{_rpad('발생', 6)}{_rpad('정상처리', 8)}"
                 f"{_rpad('실패', 6)}{_rpad('성공률', 9)}")
    lines.append("-" * 56)
    for key, label in EVENT_LABELS:
        c = counts[key]
        lines.append(f"  {_pad(label, 18)}{c['total']:>6}{c['success']:>8}{c['fail']:>6}{rate(c):>9}")
    lines.append("-" * 56)
    lines.append(f"  {_pad('전체 정지 필요', 18)}{total['total']:>6}{total['success']:>8}"
                 f"{total['fail']:>6}{rate(total):>9}")
    lines.append("=" * 56)
    return "\n".join(lines)


LIVE_W = 60


def _mmss(seconds: float) -> str:
    """초를 '분초'로 표기 (예: 192.4 → '3분 12초')."""
    total = int(round(seconds))
    return f"{total // 60}분 {total % 60:02d}초"


def _render_live(
    evaluator: "ScenarioEvaluator", sim_time: float, duration_s: float,
    npc_vehicles: int, n_peds: int,
) -> None:
    """hero.py 대시보드처럼 화면을 지우고 실시간 이벤트 표 + 시뮬 시간을 다시 그린다."""
    counts = evaluator.counts()
    active = evaluator.active_counts()

    remaining = max(0.0, duration_s - sim_time)
    frac = min(1.0, sim_time / duration_s) if duration_s > 0 else 0.0
    filled = int(round(frac * 30))
    bar = "█" * filled + "░" * (30 - filled)

    def rate(c: dict[str, int]) -> str:
        return f"{100.0 * c['success'] / c['total']:.1f}%" if c["total"] else "  -  "

    total = {"total": 0, "success": 0, "fail": 0}
    for key, _ in EVENT_LABELS:
        for f in total:
            total[f] += counts[key][f]

    lines: list[str] = []
    lines.append("═" * LIVE_W)
    lines.append("  시나리오 기반 정지-필요 이벤트 평가 (실시간)")
    lines.append(f"  NPC 차량 {npc_vehicles}대 · 보행자 {n_peds}명")
    lines.append("─" * LIVE_W)
    lines.append(f"  시뮬 시간  {_mmss(sim_time)} / {_mmss(duration_s)}"
                 f"   (남은 {_mmss(remaining)})")
    lines.append(f"  [{bar}] {100.0 * frac:5.1f}%")
    lines.append(f"  주행거리   {evaluator.distance_m / 1000:6.2f} km")
    lines.append("─" * LIVE_W)
    lines.append(f"  {_pad('이벤트', 16)}{_rpad('발생', 6)}{_rpad('정상', 6)}"
                 f"{_rpad('실패', 6)}{_rpad('진행중', 8)}{_rpad('성공률', 9)}")
    lines.append("─" * LIVE_W)
    for key, label in EVENT_LABELS:
        c = counts[key]
        lines.append(f"  {_pad(label, 16)}{c['total']:>6}{c['success']:>6}"
                     f"{c['fail']:>6}{active[key]:>6}{rate(c):>9}")
    lines.append("─" * LIVE_W)
    lines.append(f"  {_pad('전체 정지 필요', 16)}{total['total']:>6}{total['success']:>6}"
                 f"{total['fail']:>6}{sum(active.values()):>6}{rate(total):>9}")
    lines.append("═" * LIVE_W)

    sys.stdout.write("\033[H\033[2J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("runs", []) if isinstance(data, dict) else list(data)
    except (json.JSONDecodeError, OSError):
        return []


def save_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))


# ── 주행 루프 (rule_based_test.run_live 구조를 유한 시간 + 평가기 계측으로 변형) ──
def run_eval(
    *,
    host: str = "127.0.0.1",
    port: int = 2000,
    hz: float = 20.0,
    uwb_range: float = 150.0,
    duration_s: float = 600.0,
    cruise_speed: float = CRUISE_SPEED_MPS,
    route_dest_min_dist: float = 30.0,
    route_min_length: float = ROUTE_LOOKAHEAD_POINTS * ROUTE_SAMPLING_RESOLUTION_M,
    route_resync_dist_m: float = ROUTE_RESYNC_DIST_M,
    initial_speed: float = 2.0,
    seed: int | None = None,
    results_json: str | Path | None = None,
    tm_sync: bool = True,
) -> dict[str, Any]:
    """한 세션(=한 시드) 주행하며 이벤트를 평가하고 결과 run dict를 반환한다."""
    sys.path.insert(0, str(ROOT / "sim_setup" / "sensor_setting"))
    import carla  # noqa: E402

    from agent.vehicle import _setup_vehicle  # noqa: E402

    if seed is not None:
        random.seed(seed)

    build_route_planner = _load_route_planner(sampling_resolution=ROUTE_SAMPLING_RESOLUTION_M)
    lane_map = get_lane_map(MAP_JSON) if MAP_JSON.exists() else None

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()

    orig_settings = world.get_settings()
    world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=1.0 / hz))

    tm = None
    tm_was_sync = None
    if tm_sync:
        # 이 스크립트가 유일한 ticker이므로, NPC 오토파일럿이 우리 tick에 맞춰 움직이도록
        # Traffic Manager도 동기 모드로 맞춘다. 종료 시 원복.
        tm = client.get_trafficmanager()
        tm_was_sync = True  # CARLA는 현재 TM 동기 상태를 조회하는 API가 없어 종료 시 async로 되돌림
        tm.set_synchronous_mode(True)

    hero, spawned = _find_or_spawn_hero(world, random_spawn=True)
    print(f"[{_ts()}] [scenario_eval] hero id={hero.id} type={hero.type_id} (spawned={spawned})")

    if initial_speed > 0:
        fwd = hero.get_transform().get_forward_vector()
        hero.set_target_velocity(carla.Vector3D(x=fwd.x * initial_speed, y=fwd.y * initial_speed, z=0.0))

    grp = build_route_planner(world)
    route_wps: list[tuple[float, float]] = []
    route_dest = None
    off_route_ticks = 0
    resync_persist_ticks = max(1, round(hz * 0.4))

    bp_lib = world.get_blueprint_library()
    mount = carla.Transform(carla.Location(z=1.5))
    uwb_bp = bp_lib.find("sensor.other.uwb")
    uwb_bp.set_attribute("max_range", str(uwb_range))
    imu = world.spawn_actor(bp_lib.find("sensor.other.imu"), mount, attach_to=hero)
    uwb = world.spawn_actor(uwb_bp, mount, attach_to=hero)
    _, state_ref, lock = _setup_vehicle(hero, uwb, imu, send=False)

    collision = EvalCollisionSensor(hero, carla)
    evaluator = ScenarioEvaluator(hero, collision)

    # 세션 시작 시점의 NPC/보행자 수(헤더용). hero 1대는 뺀다.
    npc_vehicles = max(0, len(world.get_actors().filter("vehicle.*")) - 1)
    n_peds = len(world.get_actors().filter("walker.pedestrian.*"))
    print(f"[{_ts()}] [scenario_eval] NPC 차량 {npc_vehicles}대 · 보행자 {n_peds}명 · "
          f"목표 주행시간 {duration_s / 60:.1f}분")

    speed_pi = SpeedPIController(kp=0.6, ki=0.5, max_accel_mss=MAX_DECEL_MSS)
    half_len_m = hero.bounding_box.extent.x
    light_stop_margin = half_len_m + 1.5

    stop_ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    stopped = {"flag": False}

    def cleanup(*_a) -> None:
        if stopped["flag"]:
            return
        stopped["flag"] = True
        collision.destroy()
        for s in (imu, uwb):
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass
        if spawned:
            try:
                hero.destroy()
            except Exception:
                pass
        if tm is not None and tm_was_sync is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception:
                pass
        world.apply_settings(orig_settings)

    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(0)))

    sim_time = 0.0
    dt = 1.0 / hz
    last_render = [0.0]

    def maybe_render() -> None:
        # hero.py 대시보드처럼 ~2 Hz(벽시계)로만 화면을 다시 그린다. 시뮬은 배속으로
        # 빠르게 흐를 수 있어 매 tick 그리면 화면이 깜빡이고 느려지므로 벽시계로 throttle.
        now = time.time()
        if now - last_render[0] >= 0.5:
            _render_live(evaluator, sim_time, duration_s, npc_vehicles, n_peds)
            last_render[0] = now

    try:
        while sim_time < duration_s:
            world.tick()
            with lock:
                x, y = state_ref["est_pos"][0], state_ref["est_pos"][1]
                yaw = state_ref["imu"]["yaw"]
                speed = state_ref["wheel"]["speed"]
                raw_received = {k: dict(p) for k, p in state_ref["received"].items()}

            if x is None:
                hero.apply_control(stop_ctrl)
                # 위치 추정 전이라도 평가(지상진실)와 시간은 진행한다.
                evaluator.tick(sim_time, world)
                sim_time += dt
                maybe_render()
                continue

            route_wps, route_dest, off_route_ticks, _ = _maintain_route(
                world, grp, carla, route_wps, route_dest, x, y, yaw, off_route_ticks,
                route_dest_min_dist=route_dest_min_dist,
                input_route_length_m=route_min_length,
                resync_dist_m=route_resync_dist_m,
                resync_persist_ticks=resync_persist_ticks,
            )
            route_ahead = route_wps[:ROUTE_LOOKAHEAD_POINTS]
            ego = {"x": x, "y": y, "yaw": yaw, "speed": speed}
            received = _sort_received(raw_received, x, y)

            ctrl = rule_based_control(
                route_ahead, ego, received, speed_pi, dt,
                lane_map=lane_map, cruise_speed=cruise_speed,
                light_stop_margin_m=light_stop_margin, ego_half_len_m=half_len_m, vehicle_gap_m=VEHICLE_GAP_M,
            )

            hero.apply_control(carla.VehicleControl(
                throttle=float(ctrl["throttle"]),
                steer=float(ctrl["steer"]),
                brake=float(ctrl["brake"]),
            ))

            evaluator.tick(sim_time, world)
            sim_time += dt
            maybe_render()
    finally:
        evaluator.finalize()
        cleanup()

    run = {
        "meta": {
            "seed": seed,
            "drive_min": round(sim_time / 60.0, 2),
            "distance_km": round(evaluator.distance_m / 1000.0, 3),
            "npc_vehicles": npc_vehicles,
            "pedestrians": n_peds,
        },
        "counts": evaluator.counts(),
    }

    if results_json is not None:
        path = Path(results_json)
        runs = load_runs(path)
        runs.append(run)
        save_runs(path, runs)
        print(f"[{_ts()}] [scenario_eval] 결과 저장: {path} (누적 {len(runs)} 세션)")
        agg_counts, agg_meta = aggregate(runs)
        print("\n" + format_table(agg_counts, agg_meta))
    else:
        counts, meta = aggregate([run])
        print("\n" + format_table(counts, meta))

    return run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carla 시나리오 기반 정지-필요 이벤트 평가")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--hz", type=float, default=20.0, help="동기 모드 tick 주파수")
    p.add_argument("--uwb-range", type=float, default=150.0)
    p.add_argument("--duration", type=float, default=600.0, help="세션 주행 시간(초). 기본 600=10분")
    p.add_argument("--cruise-speed", type=float, default=CRUISE_SPEED_MPS)
    p.add_argument("--route-dest-min-dist", type=float, default=30.0)
    p.add_argument("--route-min-length", type=float, default=ROUTE_LOOKAHEAD_POINTS * ROUTE_SAMPLING_RESOLUTION_M)
    p.add_argument("--route-resync-dist-m", type=float, default=ROUTE_RESYNC_DIST_M)
    p.add_argument("--initial-speed", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None, help="hero 스폰/목적지 재현용 랜덤 시드")
    p.add_argument("--results-json", default="test/scenario_eval_results.json",
                   help="세션 결과 누적 JSON. 매 실행 끝에 전체 시드 집계 표를 출력")
    p.add_argument("--no-tm-sync", action="store_true",
                   help="Traffic Manager를 동기 모드로 맞추지 않음(NPC를 다른 프로세스가 tick할 때)")
    p.add_argument("--report", action="store_true",
                   help="주행하지 않고 --results-json의 누적 집계 표만 출력하고 종료")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.report:
        runs = load_runs(Path(args.results_json))
        if not runs:
            print(f"[scenario_eval] {args.results_json} 에 저장된 세션이 없습니다.")
            sys.exit(0)
        counts, meta = aggregate(runs)
        print(format_table(counts, meta))
        sys.exit(0)

    run_eval(
        host=args.host,
        port=args.port,
        hz=args.hz,
        uwb_range=args.uwb_range,
        duration_s=args.duration,
        cruise_speed=args.cruise_speed,
        route_dest_min_dist=args.route_dest_min_dist,
        route_min_length=args.route_min_length,
        route_resync_dist_m=args.route_resync_dist_m,
        initial_speed=args.initial_speed,
        seed=args.seed,
        results_json=args.results_json,
        tm_sync=not args.no_tm_sync,
    )

"""Town10HD 맵 로드."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import MAP_JSON


@dataclass(frozen=True)
class LanePolyline:
    road_id: int
    section_id: int
    lane_id: int
    points: np.ndarray
    widths: np.ndarray


class LaneMap:
    def __init__(
        self,
        lanes: list[LanePolyline],
        map_name: str,
        interval_m: float,
        stop_by_lane_key: dict[tuple[int, int, int], tuple[int, float, float]] | None = None,
    ):
        self.map_name = map_name
        self.interval_m = interval_m
        self.lanes = lanes
        # (road_id, section_id, lane_id) -> (tl_id, stop_x, stop_y). map_downloader.py가
        # TrafficLight.get_stop_waypoints()로 뽑은 실제 정지선 - 구버전 map JSON(필드 없음)에서는
        # 빈 dict가 되어 ego_traffic_light_stop이 항상 None을 반환한다(신호등 근사 없이 skip).
        self.stop_by_lane_key = stop_by_lane_key or {}
        self._centroids = np.array([lane.points.mean(axis=0) for lane in lanes], dtype=np.float32)
        self._lane_by_key = {(lane.road_id, lane.section_id, lane.lane_id): lane for lane in lanes}
        # (x, y, radius_m, heading_rad) -> (lane_key, proj_x, proj_y) | None — 가장 가까운
        # 차선 탐색은 좌표(+방향)만으로 결정되는 비싼 연산(segment projection)이라 캐싱한다.
        self._nearest_cache: dict[
            tuple[float, float, float, float | None], tuple[tuple[int, int, int], float, float] | None
        ] = {}
        # (x, y, lane_key) -> (proj_x, proj_y) | None — 특정 차선 위로의 투영도 좌표+차선
        # 조합 단위로 캐싱(순서 의존적인 lock 상태와 무관하게 재사용 가능).
        self._project_cache: dict[tuple[float, float, tuple[int, int, int]], tuple[float, float] | None] = {}

    @classmethod
    def load(cls, json_path: str | Path) -> LaneMap:
        json_path = Path(json_path)
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        grouped: dict[tuple[int, int, int], list[dict]] = {}
        for wp in data["waypoints"]:
            key = (int(wp["road_id"]), int(wp["section_id"]), int(wp["lane_id"]))
            grouped.setdefault(key, []).append(wp)
        lanes: list[LanePolyline] = []
        for (road_id, section_id, lane_id), wps in grouped.items():
            wps.sort(key=lambda w: float(w.get("s", 0.0)))
            points = np.array([[w["x"], w["y"]] for w in wps], dtype=np.float32)
            widths = np.array([float(w.get("lane_width", 3.5)) for w in wps], dtype=np.float32)
            if points.shape[0] >= 2:
                lanes.append(LanePolyline(road_id, section_id, lane_id, points, widths))

        stop_by_lane_key: dict[tuple[int, int, int], tuple[int, float, float]] = {}
        for tl in data.get("traffic_lights", []):
            for sw in tl.get("stop_waypoints", []):
                lane_key = (int(sw["road_id"]), int(sw["section_id"]), int(sw["lane_id"]))
                stop_by_lane_key[lane_key] = (int(tl["id"]), float(sw["x"]), float(sw["y"]))

        return cls(
            lanes=lanes,
            map_name=str(data.get("map_name", json_path.stem)),
            interval_m=float(data.get("interval_m", 2.0)),
            stop_by_lane_key=stop_by_lane_key,
        )

    def nearby_lanes(self, ego_x: float, ego_y: float, radius_m: float = 150.0, max_lanes: int = 48) -> list[LanePolyline]:
        if not self.lanes:
            return []
        dx = self._centroids[:, 0] - ego_x
        dy = self._centroids[:, 1] - ego_y
        centroid_dist = np.sqrt(dx * dx + dy * dy)
        candidates = np.where(centroid_dist <= radius_m * 1.5)[0]
        if candidates.size == 0:
            candidates = np.argsort(centroid_dist)[: max_lanes * 2]
        scored: list[tuple[float, LanePolyline]] = []
        for idx in candidates:
            lane = self.lanes[int(idx)]
            dist = float(np.sqrt((lane.points[:, 0] - ego_x) ** 2 + (lane.points[:, 1] - ego_y) ** 2).min())
            if dist <= radius_m:
                scored.append((dist, lane))
        scored.sort(key=lambda x: x[0])
        if not scored:
            scored = sorted(
                ((float(centroid_dist[i]), self.lanes[int(i)]) for i in candidates),
                key=lambda x: x[0],
            )
        return [lane for _, lane in scored[:max_lanes]]

    def _nearest_lane(
        self,
        x: float,
        y: float,
        radius_m: float,
        heading_rad: float | None = None,
        max_heading_diff_deg: float = 60.0,
    ) -> tuple[tuple[int, int, int], float, float] | None:
        """(x, y)에서 가장 가까운 차선 하나의 (lane_key, 투영 x, 투영 y). radius_m 내에 없으면 None.

        heading_rad가 주어지면, 국소 진행방향(segment tangent)이 heading_rad와
        max_heading_diff_deg 이내로 맞는 세그먼트만 후보로 본다 — 순수 좌표 거리만 쓰면
        중앙분리대 없는 도로에서 반대 방향 차선이 기하학적으로 더 가까워 거기로 스냅될 수
        있는데(ego_traffic_light_stop이 반대편 신호등을 잘못 매칭하던 것과 동일한 원인),
        heading_rad=None이면(방향을 아직 모르는 첫 지점 등) 기존처럼 방향 무시하고 고른다.
        """
        key = (x, y, radius_m, heading_rad)
        if key in self._nearest_cache:
            return self._nearest_cache[key]

        best_key = None
        best_pt = None
        best_d_sq = math.inf
        for lane in self.nearby_lanes(x, y, radius_m=radius_m, max_lanes=8):
            pts = lane.points
            for i in range(pts.shape[0] - 1):
                cx, cy = _closest_point_on_segment(
                    x, y, float(pts[i, 0]), float(pts[i, 1]), float(pts[i + 1, 0]), float(pts[i + 1, 1])
                )
                d_sq = (cx - x) ** 2 + (cy - y) ** 2
                if d_sq >= best_d_sq:
                    continue
                if heading_rad is not None:
                    tangent_yaw = math.atan2(
                        float(pts[i + 1, 1]) - float(pts[i, 1]), float(pts[i + 1, 0]) - float(pts[i, 0])
                    )
                    heading_diff = math.atan2(
                        math.sin(tangent_yaw - heading_rad), math.cos(tangent_yaw - heading_rad)
                    )
                    if abs(heading_diff) > math.radians(max_heading_diff_deg):
                        continue
                best_d_sq = d_sq
                best_pt = (cx, cy)
                best_key = (lane.road_id, lane.section_id, lane.lane_id)

        result = (best_key, best_pt[0], best_pt[1]) if best_key is not None else None
        self._nearest_cache[key] = result
        return result

    def project_onto_lane(
        self, x: float, y: float, lane_key: tuple[int, int, int]
    ) -> tuple[float, float] | None:
        """(x, y)를 지정된 lane_key 차선 위로 투영. 그 차선이 없으면 None."""
        cache_key = (x, y, lane_key)
        if cache_key in self._project_cache:
            return self._project_cache[cache_key]

        lane = self._lane_by_key.get(lane_key)
        result = None
        if lane is not None:
            pts = lane.points
            best_d_sq = math.inf
            for i in range(pts.shape[0] - 1):
                cx, cy = _closest_point_on_segment(
                    x, y, float(pts[i, 0]), float(pts[i, 1]), float(pts[i + 1, 0]), float(pts[i + 1, 1])
                )
                d_sq = (cx - x) ** 2 + (cy - y) ** 2
                if d_sq < best_d_sq:
                    best_d_sq = d_sq
                    result = (cx, cy)

        self._project_cache[cache_key] = result
        return result

    def ego_traffic_light_stop(
        self,
        x: float,
        y: float,
        ego_yaw_deg: float,
        radius_m: float = 6.0,
        max_heading_diff_deg: float = 60.0,
    ) -> tuple[int, float, float] | None:
        """(x, y)가 속한 "같은 방향" 차선에 실제로 연결된 신호등의 (tl_id, 정지선 x, 정지선 y).

        해당 차선에 연결된 신호등이 없거나(교차로가 아니거나, map JSON에 stop_waypoints가
        없는 구버전이거나) radius_m 내에 차선 자체가 없으면 None — 이 경우 호출부는 근사
        없이 신호등 feature를 아예 건너뛴다.

        순수 좌표 거리로만 "가장 가까운 차선 하나"를 고르면, 중앙분리대 없는 도로나
        교차로 근처에서 반대 방향 차선(반대편 신호등)이 기하학적으로 더 가까울 수 있다.
        그래서 반경 내 후보 차선들을 모두 훑어서, 국소 진행방향(tangent)이 ego_yaw_deg와
        max_heading_diff_deg 이내로 맞는 차선들 중 가장 가까운 것만 채택한다 — 가장 가까운
        차선이 반대 방향이어도, 그보다 조금 멀지만 같은 방향인 차선이 있으면 그걸 쓴다.
        """
        if not self.stop_by_lane_key:
            return None
        ego_yaw_rad = math.radians(ego_yaw_deg)

        best_lane_key = None
        best_d_sq = math.inf
        for lane in self.nearby_lanes(x, y, radius_m=radius_m, max_lanes=8):
            pts = lane.points
            lane_best_i, lane_best_d_sq = 0, math.inf
            for i in range(pts.shape[0] - 1):
                cx, cy = _closest_point_on_segment(
                    x, y, float(pts[i, 0]), float(pts[i, 1]), float(pts[i + 1, 0]), float(pts[i + 1, 1])
                )
                d_sq = (cx - x) ** 2 + (cy - y) ** 2
                if d_sq < lane_best_d_sq:
                    lane_best_d_sq, lane_best_i = d_sq, i
            if lane_best_d_sq > radius_m * radius_m or lane_best_d_sq >= best_d_sq:
                continue
            tangent_yaw = math.atan2(
                float(pts[lane_best_i + 1, 1]) - float(pts[lane_best_i, 1]),
                float(pts[lane_best_i + 1, 0]) - float(pts[lane_best_i, 0]),
            )
            heading_diff = math.atan2(
                math.sin(tangent_yaw - ego_yaw_rad), math.cos(tangent_yaw - ego_yaw_rad)
            )
            if abs(heading_diff) > math.radians(max_heading_diff_deg):
                continue
            best_d_sq = lane_best_d_sq
            best_lane_key = (lane.road_id, lane.section_id, lane.lane_id)

        if best_lane_key is None:
            return None
        return self.stop_by_lane_key.get(best_lane_key)

    def snap_route_sequence(
        self,
        points: list[tuple[float, float]],
        radius_m: float = 6.0,
        min_lane_hold_m: float = 10.0,
    ) -> list[tuple[float, float]]:
        """route 포인트를 순서대로 차선 중심선에 스냅하되, 짧은 거리 동안의 차선 전환은 무시한다.

        점마다 독립적으로 가장 가까운 차선에 스냅하면(구 nearest_lane_point), 실제 차선
        변경이 아니라 GT 위치 오차/커브 구간에서 순간적으로 옆 차선이 더 가까워지는 것만으로도
        route가 다른 차선으로 튀었다가 돌아온다. 이를 막기 위해 한 번 고정된 차선(locked_key)은
        다른 차선이 min_lane_hold_m 이상 이어서 더 가까울 때만 전환하고, 그 전까지는 계속
        같은 차선(앞/뒤 방향) 위로 투영한다.

        각 지점의 진행방향(직전 점 대비 heading)을 추정해 _nearest_lane에 넘긴다 — 이게
        없으면 중앙분리대 없는 도로에서 반대 방향 차선이 기하학적으로 더 가까울 때 route
        자체가 오차선(반대 차선)으로 스냅되어, 학습이 그 지점에서 반대 차선으로 향하는
        궤적을 정답으로 배우게 된다. 정지 등으로 이동량이 거의 없는 구간(< 0.3m)에서는
        heading 추정이 노이즈만 커지므로 직전 유효 heading을 그대로 유지한다.
        """
        if not points:
            return []

        result: list[tuple[float, float]] = []
        locked_key: tuple[int, int, int] | None = None
        pending_key: tuple[int, int, int] | None = None
        pending_dist = 0.0
        prev_x, prev_y = points[0]
        heading: float | None = None
        min_heading_step_m = 0.3

        for x, y in points:
            step = math.hypot(x - prev_x, y - prev_y)
            if step >= min_heading_step_m:
                heading = math.atan2(y - prev_y, x - prev_x)
            prev_x, prev_y = x, y

            nearest = self._nearest_lane(x, y, radius_m, heading_rad=heading)
            nearest_key = nearest[0] if nearest is not None else None

            if locked_key is None:
                locked_key = nearest_key
                pending_key, pending_dist = None, 0.0
            elif nearest_key is None or nearest_key == locked_key:
                pending_key, pending_dist = None, 0.0
            else:
                pending_dist = pending_dist + step if nearest_key == pending_key else step
                pending_key = nearest_key
                if pending_dist >= min_lane_hold_m:
                    locked_key = pending_key
                    pending_key, pending_dist = None, 0.0

            proj = self.project_onto_lane(x, y, locked_key) if locked_key is not None else None
            result.append(proj if proj is not None else (x, y))

        return result


def _closest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float]:
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-9:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby


_cache: dict[str, LaneMap] = {}


def get_lane_map(json_path: str | Path | None = None) -> LaneMap:
    path = Path(json_path) if json_path else MAP_JSON
    key = str(path.resolve())
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"Map JSON not found: {path}")
        _cache[key] = LaneMap.load(path)
    return _cache[key]

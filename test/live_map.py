"""Carla live 모드용 macro-map 뷰.

시뮬 화면을 직접 보기 어려운 환경(headless 서버, 원격 접속)에서 주행 상황을 확인하기 위해,
map_downloader.py가 뽑아둔 도로망(JSON) 위에 driven trajectory + 남은 route + 현재 위치를
겹쳐서 PNG 한 장으로 계속 갱신한다. 이미지 뷰어를 auto-reload로 열어두면 계속 업데이트되는
맵처럼 볼 수 있다.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
ROAD_COLOR = "#c3c2b7"
COLOR_DRIVEN = "#2a78d6"
COLOR_ROUTE = "#008300"
BG = "#fcfcfb"


class LiveMapPlotter:
    """도로망 배경은 한 번만 그리고, tick마다 driven/route/ego만 갱신해서 PNG로 저장.

    매 tick update()를 호출해도 되지만 실제 redraw/저장은 update_every tick마다만 수행
    (매 tick savefig하면 고주파(10Hz+)에서 메인 루프가 느려짐).
    """

    def __init__(
        self,
        map_json: str | Path,
        output_path: str | Path,
        update_every: int = 10,
        figsize: tuple[float, float] = (9, 9),
        dpi: int = 110,
        margin: float = 20.0,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.update_every = max(1, update_every)
        self.margin = margin
        self._tick = 0
        self._driven_x: list[float] = []
        self._driven_y: list[float] = []

        # savefig(bbox_inches="tight", dpi=110)는 한 번에 100~300ms가 걸려, 이걸 매 update_every
        # tick마다 메인 주행 루프에서 직접 부르면 그 프레임만 FPS가 뚝(60→5) 떨어진다. 그래서
        # 그림 그리기/저장은 전용 워커 스레드 하나에서만 하고, update()는 최신 스냅샷만 넘기고
        # 즉시 반환한다. matplotlib figure/artist는 이 워커 스레드에서만 건드려 thread-safe하다.
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: dict | None = None
        self._stop = False
        self._worker = threading.Thread(target=self._render_loop, name="live-map", daemon=True)

        with open(map_json, encoding="utf-8") as f:
            payload = json.load(f)

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(BG)
        self._draw_roads(payload.get("lanes", []))

        (self._driven_line,) = self.ax.plot(
            [], [], color=COLOR_DRIVEN, linewidth=2.2, zorder=4, label="driven"
        )
        (self._route_line,) = self.ax.plot(
            [], [], color=COLOR_ROUTE, linewidth=2.0, linestyle=(0, (5, 3)), zorder=3, label="route ahead"
        )
        self._ego_marker = self.ax.scatter(
            [], [], s=170, color=INK_PRIMARY, marker="^", zorder=6, edgecolors="white", linewidths=1.0, label="ego"
        )

        self.ax.set_aspect("equal")
        self.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(ROAD_COLOR)
        self.ax.set_title("Carla live — driven trajectory on map", color=INK_PRIMARY, fontsize=11, loc="left")
        self.ax.legend(fontsize=8, framealpha=0.92, loc="upper right")

        self._worker.start()

    def _draw_roads(self, lanes: list[dict]) -> None:
        for lane in lanes:
            if lane.get("lane_type") != "driving":
                continue
            pts = lane["centerline"]
            if len(pts) < 2:
                continue
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            self.ax.plot(xs, ys, color=ROAD_COLOR, linewidth=3.0, alpha=0.7, zorder=1, solid_capstyle="round")

    def update(
        self,
        ego_xy: tuple[float, float],
        route_pts: list[tuple[float, float]] | None = None,
        ego_yaw_deg: float | None = None,
    ) -> None:
        """매 tick 호출 — 값싼 작업(driven 궤적 append + 최신 스냅샷 저장)만 하고 즉시 반환.

        update_every마다 최신 상태를 워커 스레드에 넘겨 실제 그리기/savefig를 시키므로,
        메인 주행 루프는 무거운 렌더링에 절대 블록되지 않는다.
        """
        self._driven_x.append(ego_xy[0])
        self._driven_y.append(ego_xy[1])
        self._tick += 1
        if self._tick % self.update_every != 0:
            return

        # 워커가 메인 루프의 append와 경쟁하지 않도록 driven 궤적은 복사해서 넘긴다(수천~수만 점
        # 복사는 μs~ms 수준이라 무시 가능). 워커는 항상 최신 스냅샷 하나만 그리며, 렌더링이
        # 밀리면 중간 프레임은 버려지고(_pending 덮어쓰기) 최신 것만 저장된다.
        snapshot = {
            "ego_xy": ego_xy,
            "route_pts": list(route_pts) if route_pts else None,
            "ego_yaw_deg": ego_yaw_deg,
            "driven_x": list(self._driven_x),
            "driven_y": list(self._driven_y),
        }
        with self._lock:
            self._pending = snapshot
        self._wake.set()

    def _render_loop(self) -> None:
        """워커 스레드: 최신 스냅샷이 올 때마다 그려서 PNG로 저장. figure/artist는 여기서만 건드림."""
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                snapshot = self._pending
                self._pending = None
            if snapshot is not None:
                try:
                    self._render_snapshot(snapshot)
                except Exception as exc:  # 렌더 오류가 주행 루프를 죽이지 않도록 격리
                    print(f"[live_map] render 실패: {exc}")

    def _render_snapshot(self, snap: dict) -> None:
        ego_xy = snap["ego_xy"]
        route_pts = snap["route_pts"]
        driven_x, driven_y = snap["driven_x"], snap["driven_y"]

        self._driven_line.set_data(driven_x, driven_y)
        if route_pts:
            self._route_line.set_data([p[0] for p in route_pts], [p[1] for p in route_pts])
        else:
            self._route_line.set_data([], [])
        self._ego_marker.set_offsets([ego_xy])

        all_x = driven_x + [p[0] for p in route_pts or []] + [ego_xy[0]]
        all_y = driven_y + [p[1] for p in route_pts or []] + [ego_xy[1]]
        self.ax.set_xlim(min(all_x) - self.margin, max(all_x) + self.margin)
        self.ax.set_ylim(min(all_y) - self.margin, max(all_y) + self.margin)

        self.fig.savefig(self.output_path, facecolor=self.fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        plt.close(self.fig)

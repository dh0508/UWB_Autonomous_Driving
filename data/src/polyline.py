"""좌표 변환 및 신호등 상태 유틸."""

from __future__ import annotations

import math

SIGNAL_MAP = {"Red": 0.0, "Yellow": 0.5, "Green": 1.0, "Unknown": 0.25}


def world_to_ego(x: float, y: float, ego_x: float, ego_y: float, ego_yaw_deg: float) -> tuple[float, float]:
    dx = x - ego_x
    dy = y - ego_y
    yaw = math.radians(ego_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy

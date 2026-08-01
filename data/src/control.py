"""목표 속도 추종 제어기."""

from __future__ import annotations

import numpy as np


class SpeedPIController:
    """target_speed 추종 PI 제어기.

    P 제어만으로는 목표에 가까워질수록 오차가 줄어 throttle이 함께 줄고, 그 throttle이
    공기저항/구름저항과 균형을 이루는 지점에서 목표보다 낮은 속도로 수렴한다(steady-state
    error). 적분항이 그 잔여 오차를 시간에 걸쳐 누적해 정상 주행에 필요한 throttle을
    스스로 만들어 내므로, 목표 속도에 정확히 수렴한다.

    - kp: 비례 게인.
    - ki: 적분 게인(1/s). 클수록 오차를 빨리 지우지만 크면 오버슈트/진동.
    - max_accel_mss: accel→throttle 정규화 기준이자 출력(가감속) 상한.
    - 안티와인드업: 적분항 기여(ki*integral)를 [-max_accel, max_accel]로 clamp하고,
      출력이 이미 포화된 상태에서 오차가 같은 방향이면 적분을 더 쌓지 않는다(조건부 적분).
      정지 후 재출발이나 급감속에서 적분이 과누적돼 반대 조작을 방해하는 것을 막는다.
    """

    def __init__(self, kp: float = 0.6, ki: float = 0.5, max_accel_mss: float = 3.0) -> None:
        self.kp = kp
        self.ki = ki
        self.max_accel_mss = max_accel_mss
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, target_speed: float, current_speed: float, dt: float) -> dict[str, float]:
        err = target_speed - current_speed
        i_max = self.max_accel_mss / self.ki if self.ki > 0 else 0.0
        candidate = float(np.clip(self._integral + err * dt, -i_max, i_max))
        accel = self.kp * err + self.ki * candidate
        # 조건부 적분: 이미 포화된 방향으로 오차가 더 밀 때만 적분 보류. 그 외에는 갱신.
        saturated = abs(accel) >= self.max_accel_mss and (err > 0) == (accel > 0)
        if not saturated:
            self._integral = candidate
        accel = float(np.clip(
            self.kp * err + self.ki * self._integral, -self.max_accel_mss, self.max_accel_mss,
        ))
        if accel >= 0:
            return {"throttle": float(np.clip(accel / self.max_accel_mss, 0.0, 1.0)), "brake": 0.0}
        return {"throttle": 0.0, "brake": float(np.clip(-accel / self.max_accel_mss, 0.0, 1.0))}

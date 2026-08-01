#!/usr/bin/env python3

# Vehicle UWB+IMU agent.
#
# 각 차량은 위치·자세·속도를 추정하고 결과를 송신.
#   IMU listen  → roll/pitch/yaw, yaw_rate (자세만 사용).
#   UWB listen  → 모든 페이로드 수신 저장, 인프라 6개로 trilateration → 위치 추정.
#                 vehicle.get_velocity() / get_acceleration() 을 휠 엔코더 값으로 가정.
#                 → 정확한 속도·가속도 송신.
#                 위치 보정: 휠 오도메트리(스칼라 속도) + IMU(헤딩)로 dead-reckoning
#                 속도벡터를 만들고, UWB trilateration 위치와 함께 KF로 융합.
#                 (state=[x,y,vx,vy], predict=속도적분, update=UWB 위치 + 오도메트리 속도)
#
#   NLOS 각도 판별 — anchor는 좌표가 알려진 고정 인프라이므로, IMU 헤딩 기준 anchor
#                    방향각(방위각)을 매 tick 계산할 수 있다. anchor별로 "이 방향에서
#                    본 레인징이 유독 잔차가 크더라"를 온라인으로 누적 학습해(장애물
#                    배치는 환경마다 다르므로 고정 임계각 대신 데이터로 학습), 그
#                    방향의 관측치는 trilateration에서 자동으로 덜 신뢰한다.
#   MAP 제약     — CARLA 도로망(world.get_map())을 HD맵처럼 이용해, KF 위치가 주행
#                    가능 차선 밖으로 크게 벗어나면(=NLOS 드리프트로 추정) 도로 쪽으로
#                    약하게 당기는 pseudo-measurement를 추가 적용한다.
#
# IMU·UWB 센서는 sensor_attach.py 가 이미 부착해 뒀다고 가정.

import argparse
import math
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import carla

_INFRA_TYPES = {'anchor', 'traffic_light'}
_MAX_ANCHORS = 6
_MIN_ANCHORS = 3

# 위치 보정 KF 노이즈 튜닝값
_Q_POS = 0.05    # 예측 단계 위치 프로세스 노이즈 (m^2/s)
_Q_VEL = 0.10    # 예측 단계 속도 프로세스 노이즈 ((m/s)^2/s)
_R_UWB_BASE = 0.50   # UWB trilateration 위치 측정 노이즈 기준값 (m^2)
_R_UWB_MAX  = 20.0   # anchor 기하/잔차가 나쁠 때 R_UWB 상한 (폭주 방지)
_R_ODOM = 0.02   # 휠 오도메트리+IMU 헤딩 기반 속도 측정 노이즈 ((m/s)^2)
_GATE_CHI2 = 9.21    # 2-DoF 카이제곱 99% 임계값 — 튄 trilateration 이상치 게이팅

# received가 UWB 범위 이탈/차단으로 이번 tick 갱신되지 않은 항목을 이 시간(sim s) 넘게
# 들고 있으면 정리한다. 갱신 없이 계속 쌓아두면, 실제로는 범위를 벗어나거나 파괴된 차량의
# 마지막 위치·속도가 그대로 "유령 차량"으로 남아 rule_based_test가 없는 선행 차량 앞에서
# 정지하는 원인이 된다. NLOS로 인한 몇 tick 정도의 일시적 드롭아웃은 통과시키도록 여유를 둔다.
_RECEIVED_STALE_S = 1.0

# NLOS 각도 프로파일 튜닝값
_ANGLE_BIN_DEG   = 15.0                       # 방위각 히스토그램 bin 폭 (deg)
_N_ANGLE_BINS    = int(360 / _ANGLE_BIN_DEG)
_NLOS_EMA_ALPHA  = 0.05                       # bin별 잔차 EMA 갱신률 (작을수록 천천히 학습)
_NLOS_RESID_SCALE = 2.0                       # 잔차(m) → effective-distance 배율 환산 스케일

# MAP 제약 튜닝값
_MAP_LANE_MARGIN = 1.3   # 이 배수 * (차선폭/2) 를 넘어서야 "도로 이탈"로 판단 (차선 내 정상 주행엔 개입 안 함)
_R_MAP_BASE      = 1.0   # 맵 제약 pseudo-measurement 기본 노이즈 (m^2) — 이탈 폭이 클수록 더 작아져 보정이 강해짐


def _multilaterate(points):
    """가중 최소자승 trilateration.

    points: (key, x, y, z, d, w_mult) 튜플 리스트. w_mult는 호출측(NLOS 각도
    프로파일)이 계산해 넘기는 가중치 배율로, 1.0=보정없음이고 anchor 방향이
    과거에 잔차가 컸던(=NLOS 의심) 각도일수록 작아진다.

    반환: (pos, quality, residuals).
      quality는 anchor 기하(조건수)와 적합 잔차로 계산한 >=1 스칼라로, 기하가
      나쁘거나(anchor 적음/일직선에 가까움) 잔차가 크면 커진다 — KF에서
      R_UWB = R_UWB_BASE * quality 로 그 순간 측정 신뢰도를 낮추는 데 쓴다.
      residuals는 입력과 같은 순서의 (key, |예측거리-측정거리|) 리스트로,
      NLOS 각도 프로파일 온라인 학습에 쓰인다.
    """
    if len(points) < _MIN_ANCHORS:
        return None, None, None
    key1, x1, y1, z1, d1, wm1 = points[0]
    A, b, w = [], [], []
    for key, xi, yi, zi, di, wm in points[1:]:
        A.append([2*(xi - x1), 2*(yi - y1), 2*(zi - z1)])
        b.append(d1**2 - di**2
                 + xi**2 - x1**2
                 + yi**2 - y1**2
                 + zi**2 - z1**2)
        # 가까운 anchor + NLOS 아닌 방향(w_mult가 1에 가까움)일수록 신뢰
        w.append(1.0 / (wm * max(di, 1.0)))

    A = np.array(A)
    b = np.array(b)
    w = np.sqrt(np.array(w))
    Aw, bw = A * w[:, None], b * w

    pos, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

    # 수평(x,y) GDOP만 본다 — anchor가 대개 비슷한 높이에 있어 z열이 0에 가깝고,
    # (x,y,z) 전체 특이값으로 조건수를 재면 z의 관측불가능성 때문에 anchor 배치와
    # 무관하게 항상 나쁘게 나온다.
    sv_xy = np.linalg.svd(Aw[:, :2], compute_uv=False)
    cond = sv_xy[0] / sv_xy[-1] if sv_xy[-1] > 1e-6 else 50.0
    resid_rel = np.linalg.norm(Aw @ pos - bw) / (np.linalg.norm(bw) + 1e-6)
    quality = max(1.0,
                  (cond / 5.0)
                  * (1.0 + 10.0 * resid_rel)
                  * (_MAX_ANCHORS / len(points)))

    residuals = []
    for key, xi, yi, zi, di, wm in points:
        pred_d = math.sqrt((xi - pos[0])**2 + (yi - pos[1])**2 + (zi - pos[2])**2)
        residuals.append((key, abs(pred_d - di)))

    return pos, quality, residuals


class _NlosAngleProfile:
    """anchor별 (헤딩 기준 방위각 → 레인징 잔차) 온라인 프로파일.

    논문(고정 임계각으로 NLOS 판별)과 달리, 장애물 배치는 환경마다 다르므로
    고정 임계각을 쓰지 않는다. 대신 매 tick trilateration이 풀린 뒤 각
    anchor의 실제 잔차를 그 순간의 방위각 bin에 누적(EMA)해, "이 anchor를
    이 방향에서 보면 잔차가 크더라"는 패턴을 스스로 학습한다. 다음 tick부터
    그 방향의 관측치는 weight_multiplier()가 자동으로 낮게 준다.
    """

    def __init__(self):
        self._resid = {}   # {anchor_key: np.ndarray[N_ANGLE_BINS], nan=미학습}

    @staticmethod
    def _bin(angle_deg):
        return int((angle_deg % 360.0) // _ANGLE_BIN_DEG)

    def weight_multiplier(self, key, angle_deg):
        table = self._resid.get(key)
        if table is None:
            return 1.0
        r = table[self._bin(angle_deg)]
        if np.isnan(r):
            return 1.0
        return 1.0 + r / _NLOS_RESID_SCALE

    def update(self, key, angle_deg, residual):
        table = self._resid.setdefault(key, np.full(_N_ANGLE_BINS, np.nan))
        b = self._bin(angle_deg)
        if np.isnan(table[b]):
            table[b] = residual
        else:
            table[b] = (1 - _NLOS_EMA_ALPHA) * table[b] + _NLOS_EMA_ALPHA * residual


def _map_correction(world_map, x, y, z):
    """MAP 제약: (x,y)가 가장 가까운 주행 가능 차선 중심선에서 차선폭 대비
    크게 벗어나 있으면(=도로 밖 = NLOS 드리프트로 추정) 도로 쪽 pseudo-measurement를
    반환한다. 정상 주행 범위(차선 내)면 개입하지 않고 (None, None)을 반환한다.
    """
    wp = world_map.get_waypoint(
        carla.Location(x=x, y=y, z=z),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return None, None

    wp_loc = wp.transform.location
    dist = math.hypot(x - wp_loc.x, y - wp_loc.y)
    limit = (wp.lane_width / 2.0) * _MAP_LANE_MARGIN
    if dist <= limit:
        return None, None

    excess = dist - limit
    r_map = max(_R_MAP_BASE / (1.0 + excess), 0.05)   # 많이 벗어날수록 더 강하게 당김
    return (wp_loc.x, wp_loc.y), r_map


class _PositionKF:
    """등속도 칼만필터: state=[x, y, vx, vy].

    predict — 이전 속도 추정치로 위치를 적분.
    update  — 오도메트리 속도(항상)와 UWB 위치(anchor 충분할 때만)를 순차 업데이트로
              분리 융합. anchor가 일시적으로 부족(<3)해도 위치가 얼어붙지 않고
              오도메트리로 dead-reckoning을 이어간다.
              R_uwb는 매 스텝 anchor 기하/잔차 품질(quality)에 따라 적응적으로
              커지고, 튄 trilateration은 Mahalanobis 게이팅으로 약하게만 반영된다.
    """

    _H_VEL = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    _H_POS = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def __init__(self):
        self.x = None   # (4,) state
        self.P = None   # (4,4) covariance

    def _update(self, H, z, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return y, S

    def step(self, odom_vxy, dt, uwb_xy=None, uwb_quality=1.0):
        if self.x is None:
            if uwb_xy is None:
                return None, None   # 첫 위치 fix가 나오기 전엔 추정치 없음
            self.x = np.array([uwb_xy[0], uwb_xy[1], odom_vxy[0], odom_vxy[1]])
            self.P = np.eye(4)
            return self.x[0], self.x[1]

        dt = max(dt, 1e-3)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        Q = np.diag([_Q_POS * dt, _Q_POS * dt, _Q_VEL * dt, _Q_VEL * dt])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self._update(self._H_VEL, np.array(odom_vxy), np.diag([_R_ODOM, _R_ODOM]))

        if uwb_xy is not None:
            z_pos = np.array(uwb_xy)
            r_uwb = min(_R_UWB_BASE * uwb_quality, _R_UWB_MAX)
            R = np.diag([r_uwb, r_uwb])

            y = z_pos - self._H_POS @ self.x
            S = self._H_POS @ self.P @ self._H_POS.T + R
            m2 = y @ np.linalg.solve(S, y)
            if m2 > _GATE_CHI2:
                R = R * (m2 / _GATE_CHI2)   # 이상치일수록 더 약하게 반영 (soft reject)

            self._update(self._H_POS, z_pos, R)

        return self.x[0], self.x[1]

    def constrain_to_map(self, xy, r):
        """MAP 제약 pseudo-measurement 업데이트 (위치만, _H_POS 사용)."""
        if self.x is None:
            return
        self._update(self._H_POS, np.array(xy), np.diag([r, r]))


def parse_args():
    p = argparse.ArgumentParser(description='Vehicle UWB+IMU agent')
    p.add_argument('--host',      default='127.0.0.1')
    p.add_argument('-p', '--port', default=2000, type=int)
    return p.parse_args()


# ── Per-vehicle setup ─────────────────────────────────────────────────────────

def _setup_vehicle(vehicle, uwb, imu, world_map=None, *, send=True):
    ext = vehicle.bounding_box.extent
    dims = {
        'height': round(2 * ext.z, 3),
        'width':  round(2 * ext.y, 3),
        'depth':  round(2 * ext.x, 3),
    }

    lock      = threading.Lock()
    imu_state = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'yaw_rate': 0.0}
    est_pos     = [None, None, None]
    received    = {}   # {key: payload} — 수신한 모든 페이로드 저장
    received_ts = {}   # {key: 마지막 수신 sim 시각} — stale 항목 정리용 (_RECEIVED_STALE_S)
    pos_kf      = _PositionKF()
    nlos_profile = _NlosAngleProfile()
    last_kf_t   = [None]   # on_uwb 는 단일 리스너 스레드에서만 호출되므로 락 불필요
    last_imu_t  = [None]   # on_imu 갱신 시각 — on_uwb에서 yaw 외삽에 사용 (락으로 보호)
    wheel_state = {
        'vel_x': 0.0, 'vel_y': 0.0, 'speed': 0.0,
        'acc_x': 0.0, 'acc_y': 0.0, 'acc_z': 0.0, 'a_mag': 0.0,
    }

    # ── IMU 콜백: 자세만 갱신 ────────────────────────────────────────────
    def on_imu(m):
        rot = m.transform.rotation
        with lock:
            imu_state['roll']     = round(rot.roll,  3)
            imu_state['pitch']    = round(rot.pitch, 3)
            imu_state['yaw']      = round(rot.yaw,   3)
            imu_state['yaw_rate'] = round(m.gyroscope.z, 4)
            last_imu_t[0] = m.timestamp

    imu.listen(on_imu)

    # ── UWB 콜백 ─────────────────────────────────────────────────────────
    def on_uwb(measurement):
        infra_pts = []
        now_ts = measurement.timestamp

        for det in measurement:
            p = det.payload
            if not p:
                continue

            key = p.get('id') or p.get('tl_id') or id(det)
            with lock:
                received[key] = p
                received_ts[key] = now_ts

            if p.get('type') in _INFRA_TYPES:
                try:
                    infra_pts.append((
                        key,
                        float(p['x']), float(p['y']), float(p['z']),
                        det.distance,
                    ))
                except (KeyError, TypeError, ValueError):
                    pass

        # 이번 tick 갱신되지 않은 채 _RECEIVED_STALE_S 넘게 방치된 항목(범위 이탈/파괴된
        # 차량 등) 정리 — 안 그러면 마지막 위치·속도가 얼어붙은 "유령"으로 계속 남는다.
        with lock:
            stale_keys = [k for k, t in received_ts.items() if now_ts - t > _RECEIVED_STALE_S]
            for k in stale_keys:
                received.pop(k, None)
                received_ts.pop(k, None)

        # CARLA API 호출은 락 밖에서 처리
        v   = vehicle.get_velocity()      # 월드 프레임 (m/s)
        a   = vehicle.get_acceleration()  # 월드 프레임 (m/s²)
        spd   = math.sqrt(v.x**2 + v.y**2)
        a_mag = math.sqrt(a.x**2 + a.y**2)

        # 시뮬레이션 타임스탬프 사용 — synchronous_mode 에서는 실제 연산 시간(모델
        # 추론 등)이 tick 사이 벽시계 시간을 왜곡시켜 time.time() 기반 dt가 부정확해짐.
        now = measurement.timestamp
        dt  = now - last_kf_t[0] if last_kf_t[0] is not None else 0.0
        last_kf_t[0] = now

        # 휠 오도메트리(스칼라 speed) + IMU(헤딩)로 dead-reckoning 속도벡터 합성.
        # imu_state['yaw']는 별도 콜백에서 갱신되어 이 uwb 샘플 시각보다 지연될 수
        # 있음 — 회전 중(yaw_rate 큼)엔 그 지연이 헤딩 오차로 바로 이어져 R_ODOM이
        # 작은 KF가 튀는 원인이 됐다. yaw_rate로 uwb 시각까지 외삽해 지연을 보정.
        # (아래 NLOS 방위각 계산에도 같은 헤딩을 재사용한다.)
        with lock:
            yaw_deg   = imu_state['yaw']
            yaw_rate  = imu_state['yaw_rate']
            t_imu     = last_imu_t[0]
        if t_imu is not None:
            yaw_deg = yaw_deg + math.degrees(yaw_rate * (now - t_imu))
        yaw_rad = math.radians(yaw_deg)
        odom_vx = spd * math.cos(yaw_rad)
        odom_vy = spd * math.sin(yaw_rad)

        # ── NLOS 각도 가중치: 직전 위치 추정치 기준 anchor 방위각을 계산해,
        # 그 방향에서 과거에 잔차가 컸던 anchor는 이번 trilateration에서 덜 신뢰한다.
        # 첫 fix가 아직 없으면(ref_xy=None) 판별 불가 → 중립 가중치(1.0)로 통과.
        ref_xy = (pos_kf.x[0], pos_kf.x[1]) if pos_kf.x is not None else None
        weighted_pts = []
        for key, xi, yi, zi, di in infra_pts:
            if ref_xy is not None:
                bearing = math.degrees(math.atan2(yi - ref_xy[1], xi - ref_xy[0])) - yaw_deg
                bearing %= 360.0
            else:
                bearing = None
            wm = nlos_profile.weight_multiplier(key, bearing) if bearing is not None else 1.0
            weighted_pts.append((key, xi, yi, zi, di, wm, bearing))

        # anchor가 3개 미만이라도(가림/일시 dropout) 오도메트리로 predict는 계속한다 —
        # 이전엔 여기서 바로 return 해서 위치가 그대로 얼어붙었다.
        pos, quality = None, None
        if len(weighted_pts) >= _MIN_ANCHORS:
            # eff_dist = 실거리 * NLOS 배율 — NLOS 의심 방향의 anchor는 더 멀리 취급해
            # top-N 선별에서도, LS 가중치에서도 자연히 밀려나게 한다.
            nearest = sorted(weighted_pts, key=lambda d: d[4] * d[5])[:_MAX_ANCHORS]
            fit_pts = [(k, x, y, z, d, wm) for k, x, y, z, d, wm, _ in nearest]
            pos, quality, residuals = _multilaterate(fit_pts)
            if residuals is not None:
                bearing_by_key = {k: b for k, _, _, _, _, _, b in nearest}
                for key, resid in residuals:
                    b = bearing_by_key.get(key)
                    if b is not None:
                        nlos_profile.update(key, b, resid)

        uwb_xy = (float(pos[0]), float(pos[1])) if pos is not None else None
        kf_x, kf_y = pos_kf.step((odom_vx, odom_vy), dt,
                                  uwb_xy=uwb_xy, uwb_quality=quality or 1.0)
        if kf_x is None:
            return   # 첫 위치 fix가 아직 없음 — 이번 tick은 보낼 추정치가 없음

        # ── MAP 제약: 도로 밖으로 크게 벗어난 추정치만 도로 쪽으로 약하게 당긴다
        # (차선 내 정상 주행 범위에서는 개입하지 않음 — _map_correction 참고).
        if world_map is not None:
            with lock:
                z_hint = est_pos[2] if est_pos[2] is not None else 0.0
            corr_xy, r_map = _map_correction(world_map, kf_x, kf_y, z_hint)
            if corr_xy is not None:
                pos_kf.constrain_to_map(corr_xy, r_map)
                kf_x, kf_y = float(pos_kf.x[0]), float(pos_kf.x[1])

        with lock:
            # est_pos 3원소를 원자적으로 갱신 (race condition 방지)
            est_pos[0] = round(float(kf_x), 2)
            est_pos[1] = round(float(kf_y), 2)
            if pos is not None:
                est_pos[2] = round(float(pos[2]), 2)   # 고도는 오도메트리로 보정할 축이 없어 그대로 사용
            s = dict(imu_state)
            wheel_state['vel_x'] = round(v.x, 3)
            wheel_state['vel_y'] = round(v.y, 3)
            wheel_state['speed'] = round(spd, 3)
            wheel_state['acc_x'] = round(a.x, 4)
            wheel_state['acc_y'] = round(a.y, 4)
            wheel_state['acc_z'] = round(a.z, 4)
            wheel_state['a_mag'] = round(a_mag, 4)

        if send:
            uwb.send_uwb_payload({
                'type':     'vehicle',
                'id':       vehicle.id,
                # 위치 — trilateration + 휠 오도메트리/IMU KF 융합 추정 (m)
                'x':        est_pos[0],
                'y':        est_pos[1],
                'z':        est_pos[2],
                # 자세 — IMU (deg)
                'roll':     s['roll'],
                'pitch':    s['pitch'],
                'yaw':      s['yaw'],
                # 속도 — 휠 엔코더 (m/s, 월드 프레임)
                'vel_x':    round(v.x, 3),
                'vel_y':    round(v.y, 3),
                'speed':    round(spd, 3),
                # 가속도 — 휠 엔코더 미분 (m/s², 월드 프레임)
                'acc_x':    round(a.x, 4),
                'acc_y':    round(a.y, 4),
                'acc_z':    round(a.z, 4),
                'a_mag':    round(a_mag, 4),
                # 조향 — IMU gyroscope z (rad/s)
                'yaw_rate': s['yaw_rate'],
                # 차량 규격 (m)
                **dims,
            })

    uwb.listen(on_uwb)

    state_ref = {
        'imu':     imu_state,
        'est_pos': est_pos,
        'received': received,
        'wheel':   wheel_state,
    }
    return [imu, uwb], state_ref, lock


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world     = client.get_world()
    world_map = world.get_map()   # MAP 제약(도로망 매칭)에 사용

    active_sensors = []

    def cleanup(sig=None, frame=None):
        print(f'\n[vehicle] Stopping {len(active_sensors)} listeners...')
        for s in reversed(active_sensors):
            try:
                if s.is_alive:
                    s.stop()
            except Exception:
                pass
        print('[vehicle] Done.')
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # sensor_attach.py 가 차량에 센서 부착을 완료할 때까지 대기.
    # UWB 개수가 2초 연속 동일하면 부착 완료로 판단.
    # generate_traffic.py 가 실행 중이 아니어도(차량 0대) 넘어갈 수 있도록,
    # 0대인 상태도 안정 상태로 인정한다.
    print('[vehicle] 차량 및 센서 대기 중 (차량 0대여도 진행)...')
    prev_uwb_cnt = -1
    stable_ticks = 0
    while stable_ticks < 2:
        vehicle_ids = {v.id for v in world.get_actors().filter('vehicle.*')}
        uwb_cnt = sum(
            1 for s in world.get_actors().filter('sensor.other.uwb')
            if s.parent is not None and s.parent.id in vehicle_ids
        )
        if uwb_cnt == prev_uwb_cnt:
            stable_ticks += 1
        else:
            stable_ticks = 0
        prev_uwb_cnt = uwb_cnt
        print(f'\r  차량={len(vehicle_ids)}  UWB={uwb_cnt} ', end='', flush=True)
        time.sleep(1.0)
    print()

    uwb_map = {}
    imu_map = {}
    for s in world.get_actors().filter('sensor.other.*'):
        if s.parent is None or s.parent.id not in vehicle_ids:
            continue
        if s.type_id == 'sensor.other.uwb':
            uwb_map[s.parent.id] = s
        elif s.type_id == 'sensor.other.imu':
            imu_map[s.parent.id] = s

    vehicles = list(world.get_actors().filter('vehicle.*'))
    print(f'[vehicle] {len(vehicles)} vehicles  '
          f'| {len(uwb_map)} UWBs  | {len(imu_map)} IMUs\n')

    ok = 0
    for v in vehicles:
        uwb = uwb_map.get(v.id)
        imu = imu_map.get(v.id)
        missing = ' '.join(
            x for x, s in [('UWB', uwb), ('IMU', imu)] if s is None
        )
        if missing:
            print(f'  vehicle {v.id:>6}: {missing} 없음 — sensor_attach.py 먼저 실행')
            continue

        sensors, _, _ = _setup_vehicle(v, uwb, imu, world_map)
        active_sensors.extend(sensors)
        ok += 1
        print(f'  vehicle {v.id:>6}  UWB={uwb.id:>8}  IMU={imu.id:>8}')

    print(f'\n  → {ok} vehicles active')
    print('[vehicle] Running... Ctrl+C to stop')

    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()

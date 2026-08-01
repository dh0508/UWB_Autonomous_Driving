#!/usr/bin/env python3

# 센서 부착 + 에이전트 순차 기동 launcher.
#
# 실행 순서:
#   1. sensor_setting/sensor_attach.py  — 차량/보행자에 IMU·UWB 부착, roadside anchor 스폰
#      (완료 로그 감지 후 자동으로 다음 단계 진행)
#   2. sensor_setting/run_agents.py     — infrastructure → vehicle → pedestrian (1초 간격)
#      (실행 확인 후 완료 메시지 출력)
#
# 전제: 없음. generate_traffic.py 는 선택 사항 — 실행 중이 아니면
#       차량/보행자 센서 부착 없이 인프라(로드사이드 앵커 + 신호등 UWB)만 설치하고 진행.
#
# 사용:
#   python sim_setup/run_setup.py [--host 127.0.0.1] [-p 2000] [--uwb-range 150]
#
# Ctrl+C → 전체 종료 (1회)

import os
import signal
import subprocess
import sys
import threading
import time

HERE      = os.path.dirname(os.path.abspath(__file__))
SETUP_DIR = os.path.join(HERE, 'sensor_setting')

SENSOR_ATTACH = os.path.join(SETUP_DIR, 'sensor_attach.py')
RUN_AGENTS    = os.path.join(SETUP_DIR, 'run_agents.py')

procs    = []
_exiting = False
_summary = {}   # 에이전트 완료 후 출력할 요약 정보


def cleanup(sig=None, frame=None):
    global _exiting
    if _exiting:
        return
    _exiting = True
    print('\n[run_setup] 종료 중...')
    for p in reversed(procs):
        if p.poll() is None:
            p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)


def _pipe_output(proc, done_event=None, markers=None, capture_summary=False):
    remaining = set(markers) if markers else set()
    for raw in iter(proc.stdout.readline, b''):
        if _exiting:
            break
        line = raw.decode(errors='replace').rstrip()
        print(line, flush=True)
        if capture_summary:
            s = line.strip()
            if s.startswith('→'):
                if 'anchor' in s:
                    _summary['anchors'] = s
                elif 'traffic light' in s:
                    _summary['tl'] = s
                elif 'vehicles active' in s:
                    _summary['vehicles'] = s
                elif 'walkers active' in s:
                    _summary['walkers'] = s
        if done_event and remaining:
            for m in list(remaining):
                if m in line:
                    remaining.discard(m)
            if not remaining:
                done_event.set()
    proc.stdout.close()


def start_and_wait(script, args, markers, step_name, capture_summary=False):
    """서브프로세스를 시작하고 markers 로그가 모두 출력될 때까지 대기."""
    done = threading.Event()
    print(f'[run_setup] {step_name} 시작...')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    proc = subprocess.Popen(
        [sys.executable, '-u', script] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Ctrl+C 가 직접 전달되지 않도록 격리
        env=env,
    )
    procs.append(proc)

    t = threading.Thread(
        target=_pipe_output,
        args=(proc, done, markers, capture_summary),
        daemon=True,
    )
    t.start()

    while not done.is_set():
        if _exiting:
            return
        if proc.poll() is not None:
            print(f'[run_setup] {step_name} 예기치 않게 종료됨.')
            cleanup()
            return
        time.sleep(0.2)

    print(f'[run_setup] {step_name} 완료. 2초 대기 후 다음 단계 진행...')
    time.sleep(2)
    print()


extra_args = sys.argv[1:]

start_and_wait(
    SENSOR_ATTACH, extra_args,
    markers=['[sensor_attach] Total sensors:'],
    step_name='sensor_attach.py',
)

start_and_wait(
    RUN_AGENTS, extra_args,
    markers=[
        '[infrastructure] Running',
        '[vehicle] Running',
        '[pedestrian] Running',
    ],
    step_name='run_agents.py',
    capture_summary=True,
)

print('[run_setup] 모든 설정이 완료되었습니다. Ctrl+C 로 종료')
print(f'  인프라:  {_summary.get("anchors", "앵커 ?")}  |  {_summary.get("tl", "신호등 ?")}')
print(f'  차량:    {_summary.get("vehicles", "?")}')
print(f'  보행자:  {_summary.get("walkers", "?")}')
print()

while True:
    for p in procs:
        rc = p.poll()
        if rc is not None:
            msg = f'비정상 종료 (return code: {rc})' if rc != 0 else '종료됨'
            print(f'[run_setup] PID {p.pid} {msg}. 전체 중단.')
            cleanup()
    time.sleep(1)

#!/usr/bin/env python3

# scenario_eval.py 를 매 회차 새 랜덤 시드로 --runs 회 반복 실행하는 배치 러너.
#
# scenario_eval.py 상단 docstring의 수동 절차(1~5)를 그대로 자동화한다:
#   1. generate_traffic.py -n <N> -w <W> -s <seed>  (새 시드로 NPC 차량/보행자 재생성)
#   2. sim_setup/run_setup.py                         (sensor_attach.py → run_agents.py
#                                                      [infrastructure/vehicle/pedestrian] 순차 기동)
#   3. scenario_eval.py --seed <seed> --duration <D> --results-json <path>  (포그라운드, 종료까지 대기)
#   4. 1~2에서 띄운 프로세스를 SIGINT로 종료(각자 cleanup에서 스폰한 액터 정리)하고 다음 회차 진행.
#
# --results-json에 매 회차 결과가 누적되고, scenario_eval.py가 매 실행 끝에 전체 시드 집계 표를
# 스스로 출력하므로 이 러너는 별도 집계를 하지 않는다.
#
# 사용:
#   python test/run_scenario_eval_seeds.py --runs 30 --duration 600
#
# Ctrl+C → 현재 회차의 백그라운드 프로세스를 정리하고 종료.

from __future__ import annotations

import argparse
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
GEN_TRAFFIC   = ROOT / "sim_setup" / "scenarios" / "generate_traffic.py"
RUN_SETUP     = ROOT / "sim_setup" / "run_setup.py"
SCENARIO_EVAL = ROOT / "test" / "scenario_eval.py"

_exiting = False
_stage_procs: list[tuple[subprocess.Popen, str]] = []


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _pipe_output(proc: subprocess.Popen, done: threading.Event, markers: set[str]) -> None:
    """자식 stdout을 그대로 중계하면서, markers가 모두 한 번씩 나오면 done을 set한다."""
    remaining = set(markers)
    for raw in iter(proc.stdout.readline, b""):
        if _exiting:
            break
        line = raw.decode(errors="replace").rstrip()
        print(f"  | {line}", flush=True)
        for m in list(remaining):
            if m in line:
                remaining.discard(m)
        if not remaining and not done.is_set():
            done.set()
    try:
        proc.stdout.close()
    except Exception:
        pass


def _start_bg(script: Path, args: list[str], markers: list[str], name: str, timeout: float) -> subprocess.Popen:
    """백그라운드 프로세스를 띄우고 markers 로그가 모두 찍힐 때까지 대기한다."""
    print(f"[{_ts()}] [batch] {name} 시작...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Ctrl+C가 직접 전달되지 않도록 격리 — 종료는 SIGINT를 명시적으로 보낸다
        env=env,
    )
    _stage_procs.append((proc, name))

    done = threading.Event()
    threading.Thread(target=_pipe_output, args=(proc, done, set(markers)), daemon=True).start()

    start = time.time()
    while not done.is_set():
        if _exiting:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"{name} 이(가) 준비 전에 종료됨 (returncode={proc.returncode})")
        if time.time() - start > timeout:
            raise RuntimeError(f"{name} 준비 대기 시간 초과({timeout:.0f}s)")
        time.sleep(0.2)
    print(f"[{_ts()}] [batch] {name} 준비 완료.")
    return proc


def _stop_bg(proc: subprocess.Popen, name: str, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    print(f"[{_ts()}] [batch] {name} 종료 중 (SIGINT)...")
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"[{_ts()}] [batch] {name} 이(가) {timeout:.0f}s 안에 종료되지 않아 강제 종료.")
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def _teardown_stage() -> None:
    for proc, name in reversed(_stage_procs):
        _stop_bg(proc, name)
    _stage_procs.clear()


def _handle_signal(sig, frame) -> None:
    global _exiting
    if _exiting:
        return
    _exiting = True
    print(f"\n[{_ts()}] [batch] 중단 요청 — 현재 회차 백그라운드 프로세스 정리 후 종료.")
    _teardown_stage()
    sys.exit(1)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def run_one(seed: int, args: argparse.Namespace) -> int:
    """시드 하나로 트래픽 생성→run_setup(센서 부착+에이전트)→평가 파이프라인을 한 번 돌린다."""
    _stage_procs.clear()

    _start_bg(
        GEN_TRAFFIC,
        ["--host", args.host, "--port", str(args.port),
         "-n", str(args.num_vehicles), "-w", str(args.num_walkers),
         "-s", str(seed), "--tm-port", str(args.tm_port)],
        markers=["spawned"],
        name="generate_traffic.py",
        timeout=args.setup_timeout,
    )
    # run_setup.py는 받은 인자를 sensor_attach.py + run_agents.py(→ infrastructure/vehicle/
    # pedestrian)에 그대로 전달하는데, vehicle.py/pedestrian.py는 --host/--port만 받고
    # --uwb-range는 모른다. 여기서 넘기면 두 에이전트가 argparse 에러로 죽고 그 여파로
    # run_agents.py → run_setup.py 전체가 조기 종료된다. 그래서 --host/--port만 넘긴다
    # (sensor_attach.py/infrastructure.py의 --uwb-range 기본값 150.0을 그대로 쓴다).
    _start_bg(
        RUN_SETUP,
        ["--host", args.host, "--port", str(args.port)],
        markers=["[run_setup] 모든 설정이 완료되었습니다"],
        name="sim_setup/run_setup.py",
        timeout=args.setup_timeout,
    )

    if _exiting:
        return -1

    print(f"[{_ts()}] [batch] 설정 완료. 2초 대기 후 scenario_eval.py 시작...")
    time.sleep(2.0)

    eval_args = [
        sys.executable, str(SCENARIO_EVAL),
        "--host", args.host,
        "--port", str(args.port),
        "--duration", str(args.duration),
        "--seed", str(seed),
        "--results-json", args.results_json,
    ]
    print(f"[{_ts()}] [batch] scenario_eval.py 실행 (seed={seed})...")
    result = subprocess.run(eval_args)

    _teardown_stage()
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scenario_eval.py를 랜덤 시드로 N회 반복 실행")
    p.add_argument("--runs", type=int, default=30, help="반복 횟수 (기본 30)")
    p.add_argument("--duration", type=float, default=600.0, help="회차당 평가 주행시간(초)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--num-vehicles", type=int, default=30)
    p.add_argument("--num-walkers", type=int, default=8)
    p.add_argument("--results-json", default="test/scenario_eval_results.json")
    p.add_argument("--setup-timeout", type=float, default=60.0, help="트래픽/센서/인프라 준비 대기 제한(초)")
    p.add_argument("--max-consecutive-failures", type=int, default=2,
                   help="연속 실패 허용 횟수 — 초과 시 CARLA 서버 문제로 보고 남은 회차를 중단(기본 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random()
    consecutive_failures = 0

    for i in range(1, args.runs + 1):
        if _exiting:
            break
        seed = rng.randint(0, 2**31 - 1)
        print(f"\n{'=' * 60}\n[{_ts()}] [batch] 회차 {i}/{args.runs}  seed={seed}\n{'=' * 60}")
        try:
            rc = run_one(seed, args)
        except Exception as e:
            print(f"[{_ts()}] [batch] 회차 {i} 실패: {e}")
            _teardown_stage()
            rc = -1

        if rc == 0:
            consecutive_failures = 0
            continue
        if _exiting:
            break

        consecutive_failures += 1
        print(f"[{_ts()}] [batch] scenario_eval.py returncode={rc} (연속 실패 {consecutive_failures}회)")
        # 연속 실패는 스폰 충돌 같은 일회성 문제가 아니라 CARLA 서버 자체가 죽었거나
        # 응답 없는 상태일 가능성이 크다. 이 경우 남은 회차를 전부 즉시 재실패시키며
        # 낭비하는 대신 조기 중단하고 사용자가 서버를 확인/재시작하게 한다.
        if consecutive_failures >= args.max_consecutive_failures:
            print(f"[{_ts()}] [batch] {consecutive_failures}회 연속 실패 — CARLA 서버 응답을 확인하세요. "
                  f"남은 회차를 중단합니다.")
            break

    print(f"\n[{_ts()}] [batch] 종료 (요청 {args.runs}회차 중 처리 완료).")


if __name__ == "__main__":
    main()

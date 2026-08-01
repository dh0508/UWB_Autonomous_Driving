#!/usr/bin/env python3

# agent/ 폴더의 세 에이전트를 동시에 실행.
# Ctrl+C 하면 전부 종료.

import os
import signal
import subprocess
import sys

HERE   = os.path.dirname(os.path.abspath(__file__))
AGENTS = [
    os.path.join(HERE, 'agent', 'infrastructure.py'),
    os.path.join(HERE, 'agent', 'vehicle.py'),
    os.path.join(HERE, 'agent', 'pedestrian.py'),
]

# infrastructure → vehicle → pedestrian 순서로 기동
# 각 에이전트는 내부 대기 루프를 가지지만, 순서를 지켜야 anchor 탐색이 안정적으로 동작
procs = []
for a in AGENTS:
    procs.append(subprocess.Popen([sys.executable, a] + sys.argv[1:]))
    __import__('time').sleep(1)

print(f'[run_agents] 실행 중: {[os.path.basename(a) for a in AGENTS]}')
print('[run_agents] Ctrl+C 로 전체 종료\n')

def cleanup(sig=None, frame=None):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

# 하나라도 죽으면 나머지도 종료
while True:
    for p in procs:
        if p.poll() is not None:
            print(f'[run_agents] PID {p.pid} 종료됨. 전체 중단.')
            cleanup()
    __import__('time').sleep(1)

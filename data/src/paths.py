"""프로젝트 경로 상수."""

from __future__ import annotations

from pathlib import Path

# UME_Drive_Net-main/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MAP_DIR = DATA_ROOT / "map"
MAP_JSON = MAP_DIR / "Town10HD_Opt.json"

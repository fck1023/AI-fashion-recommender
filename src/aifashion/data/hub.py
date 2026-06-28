"""雙雲 artifact pipeline:Drive(快取)+ HF Hub(版本化 Source-of-Truth)。

來源:mater_kai_datascienctist.py(打 TAR、items.csv/paths.json、push 到 HF、
     verify_report.json)+ 處理資料集part2.py。

設計重點(這是面試第二個主打 — feature store / model registry 的 pattern):
  - Drive = 可變的快取/工作區(下載、暫存、續傳);HF Hub = 不可變、可回溯的 SoT。
  - 發佈流程:產 artifacts → 版本化上傳 → 自動驗證 → 更新 `latest` 別名。
  - 用 `latest` 別名「解耦產出者與消費者」:demo/eval/API 永遠只讀 `latest`。
  - 下載維持「可續傳、冪等(只補缺的前綴直到缺口為 0)」。
"""
from __future__ import annotations

from pathlib import Path


def publish_artifacts(run_dir: Path, repo_id: str, version: str) -> str:
    """把 vecs/index/keys/meta/stats 版本化上傳 HF Hub,回傳 commit/tag。"""
    raise NotImplementedError("Phase 1：自 mater_kai 的 push 區段搬入")


def verify_artifacts(run_dir: Path) -> dict:
    """自動驗證(抽樣開圖、尺寸、壞檔/重複),回傳 verify_report。"""
    raise NotImplementedError("Phase 1")


def update_latest_alias(repo_id: str, version: str) -> None:
    """把 `latest` 指到指定版本,讓下游無痛切換。"""
    raise NotImplementedError("Phase 1")

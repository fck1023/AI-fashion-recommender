"""資料準備:Fashionpedia → 清理 → WebDataset 分片。

來源(待 Phase 1 重構、去重):
  - mater_kai_datascienctist.py(Route A:批次下載到 Drive、打 TAR、items.csv/paths.json)
  - 處理資料集part2.py(抽樣、CLIP-L/14-336 抽向量)

重構重點:
  - 把重複 N 次的 norm_rel / rsync_prefix / ImgDS / encode_images 收斂成單一實作。
  - 批次下載維持「可續傳、冪等(只補缺的前綴直到缺口為 0)」的韌性設計。
"""
from __future__ import annotations

from pathlib import Path


def build_shards(raw_dir: Path, out_dir: Path, shard_size: int = 1000) -> Path:
    """把清理過的影像打包成 WebDataset 分片(.tar)+ manifest.json。"""
    raise NotImplementedError("Phase 1：自 mater_kai_datascienctist.py 搬入並去重")


def normalize_rel(path: str) -> str:
    """把資料集內的相對路徑正規化(原始碼裡被重複定義了 5 次)。"""
    raise NotImplementedError("Phase 1")

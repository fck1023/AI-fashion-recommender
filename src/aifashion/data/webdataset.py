"""訓練用的 robust WebDataset 串流(Drive 優先,HF 保底)。

來源:ai_fashion_..._blip_v7.py Cell 3-4(GenDataset / make_wds_robust /
     collate_batch / make_loader / list_shards / copy_to_ssd /
     build_streams_from_drive / build_streams_from_hf)。

設計重點:
  - 大資料不全載進記憶體,改用「分片 .tar + 串流」逐批讀。
  - 「robust」= 單一壞樣本/壞檔不讓整個 epoch 掛掉(逐樣本 try/except)。
  - Drive 找不到 shards 時自動回退到 Hugging Face 資料集。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def list_shards(shards_dir: Path, limit: int = 0) -> list[str]:
    """列出 .tar 分片 URL(limit=0 表示全部)。"""
    raise NotImplementedError("Phase 1：自 v7 Cell 4 搬入")


def build_streams(prefer: str = "drive") -> tuple:
    """建立 (train_stream, val_stream);drive 失敗自動回退 hf。"""
    raise NotImplementedError("Phase 1")


def make_loader(stream_fn: Callable, preprocess, tokenizer, batch_size: int):
    """把串流包成 DataLoader(含 collate)。"""
    raise NotImplementedError("Phase 1")

"""訓練用的 robust WebDataset 串流(Drive 優先,HF 保底)。

來源:ai_fashion_..._blip_v7.py
  - Cell 3(行 212–236):GenDataset / make_wds_robust / collate_batch / make_loader
  - Cell 4(行 238–300):list_shards / copy_to_ssd / build_streams_from_drive / _from_hf

──────────────────────────────────────────────────────────────────────────
設計重點(面試可講)
──────────────────────────────────────────────────────────────────────────
  - 大資料不全載進記憶體:分片 .tar + 串流逐批讀(WebDataset)。
  - 「robust」= 單一壞檔/壞樣本不讓整個 epoch 掛掉:逐樣本解碼,失敗就 select 掉。
  - 雙來源:Drive 找不到 shards 時自動回退到 Hugging Face 資料集(對應 hub.py 的 SoT)。
  - 關注點分離:本模組只管「讀圖 + 組 batch」;caption 文字監督交給 captions.resolve_caption,
    所以 cap_map(BLIP 強化 caption)能在串流當下就生效。

loader contract:每個 batch 產出 (images, tokens, captions, keys) —— train.py / eval.py 即依此消費。
"""
from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Callable

from ..configs import CFG
from .captions import resolve_caption


# ── 單樣本解碼:讀圖(本模組職責)+ 解析 caption(委派 captions)────────────
def _decode_sample(sample: dict, cap_map: dict[str, str] | None = None):
    """從 WDS raw sample 解出 {image, caption, key};壞檔或噪聲 caption 回 None(丟棄)。"""
    from PIL import Image

    img = None
    for k in CFG.data.img_keys:
        if k not in sample:
            continue
        v = sample[k]
        try:  # 改進:原 notebook 沒包 try/except,一張壞 jpg 會讓整個 epoch 掛掉
            if isinstance(v, (bytes, bytearray)):
                img = Image.open(io.BytesIO(v)).convert("RGB")
            elif hasattr(v, "convert"):
                img = v.convert("RGB")
            else:
                img = Image.open(v).convert("RGB")
        except Exception:
            return None
        break
    if img is None:
        return None

    cap = resolve_caption(sample, cap_map)
    if cap is None:                          # 噪聲/過短 caption → 丟棄
        return None
    return {"image": img, "caption": cap, "key": sample.get("__key__", "")}


def make_wds_robust(urls: list[str], cap_map: dict[str, str] | None = None) -> Callable:
    """建立逐樣本容錯的 WDS 串流;回傳 generator function,yield (PIL, caption, key)。"""
    import webdataset as wds

    ds = (wds.WebDataset(urls, resampled=False)
          .map(lambda s: _decode_sample(s, cap_map))
          .select(lambda x: x is not None))

    def gen():
        for x in ds:
            yield x["image"], x["caption"], x["key"]
    return gen


def _gen_dataset(gen_fn: Callable):
    """把 generator function 包成 torch IterableDataset(延遲 import torch)。"""
    from torch.utils.data import IterableDataset

    class _GenDataset(IterableDataset):
        def __iter__(self):
            yield from gen_fn()
    return _GenDataset()


def collate_batch(batch, preprocess, tokenizer):
    """把 [(PIL, caption, key), ...] collate 成 (images, tokens, captions, keys)。"""
    import torch

    imgs, caps, keys = zip(*batch)
    images = torch.stack([preprocess(im) for im in imgs], 0)
    tokens = tokenizer(list(caps))
    return images, tokens, list(caps), list(keys)


def make_loader(stream_fn: Callable, preprocess, tokenizer,
                batch_size: int | None = None, *, num_workers: int = 0):
    """把串流包成 DataLoader(含 collate)。batch_size 預設取 configs.TrainConfig。"""
    from torch.utils.data import DataLoader

    batch_size = CFG.train.batch_size if batch_size is None else batch_size
    return DataLoader(
        _gen_dataset(stream_fn), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_batch(b, preprocess, tokenizer),
        drop_last=False, pin_memory=True,
    )


# ── shards 來源:本地 / Drive ─────────────────────────────────────────────
def list_shards(shards_dir=None, limit: int | None = None) -> list[str]:
    """列出 .tar 分片路徑(limit>0 只取前 N 個;預設取 configs.DataConfig)。"""
    shards_dir = Path(shards_dir or CFG.data.shards_dir)
    limit = CFG.data.shards_limit if limit is None else limit
    if not shards_dir.is_dir():
        return []
    urls = sorted(str(p) for p in shards_dir.glob("*.tar"))
    return urls[:limit] if limit and limit > 0 else urls


def copy_to_ssd(urls: list[str], ssd_dir: str = "/content/wds_cache") -> list[str]:
    """把 shards rsync 到本機 SSD(Colab 上比 Drive I/O 快);冪等:大小相同就跳過。"""
    if not urls:
        return []
    ssd = Path(ssd_dir)
    ssd.mkdir(parents=True, exist_ok=True)
    local = []
    for u in urls:
        dst = ssd / Path(u).name
        if not dst.exists() or os.path.getsize(u) != os.path.getsize(dst):
            subprocess.call(["rsync", "-a", str(u), str(dst)])
        local.append(str(dst))
    return local


# ── 建立 train/val 串流(Drive 優先,HF 保底)────────────────────────────
def build_streams_from_drive(*, copy_ssd: bool = False,
                             cap_map: dict[str, str] | None = None):
    """從本地/Drive 的 .tar shards 建串流;找不到回 (None, None)。"""
    urls = list_shards()
    if not urls:
        return None, None
    if copy_ssd:
        urls = copy_to_ssd(urls)
    print(f"找到 {len(urls)} 個 Drive shards。")
    # 無獨立 val shards 時先共用同一批(沿用原 notebook 行為)
    return make_wds_robust(urls, cap_map), make_wds_robust(urls, cap_map)


def build_streams_from_hf():
    """保底來源:從 HF 資料集(非串流、本地快取)建 (train, val) 串流。"""
    from datasets import load_dataset
    d = CFG.data

    def _load(split: str, fallback: str):
        try:
            return load_dataset(d.hf_repo_id, split=split, streaming=False)
        except Exception:
            return load_dataset(d.hf_repo_id, split=fallback, streaming=False)

    ds_tr = _load(d.hf_train_slice, "train[:10%]")
    ds_va = _load(d.hf_val_slice, "train[:2%]")

    def iter_hf(ds):
        for x in ds:
            im = x.get("image")
            if im is None:
                continue
            cap = x.get("caption") or x.get("text") or d.default_caption
            yield im.convert("RGB"), str(cap), ""    # HF 路徑無 key

    print("使用 HF 本地快取資料集(保底)。")
    return (lambda: iter_hf(ds_tr)), (lambda: iter_hf(ds_va))


def build_streams(prefer: str = "drive", *, copy_ssd: bool = False,
                  cap_map: dict[str, str] | None = None) -> tuple:
    """建立 (train_stream, val_stream, source);prefer='drive' 找不到 shards 時自動回退 HF。

    要套用 BLIP 強化 caption:先 `cap_map = captions.load_cap_map(run_dir/'cap_map.jsonl')`
    再傳進來,串流當下就會用覆寫鏈蓋掉弱啟發式。
    """
    if prefer == "drive":
        train, val = build_streams_from_drive(copy_ssd=copy_ssd, cap_map=cap_map)
        if train is not None:
            return train, val, "DRIVE_WDS"
        print("⚠️ Drive shards 未找到,改用 HF 保底。")
    train, val = build_streams_from_hf()
    return train, val, "HF_LOCAL"

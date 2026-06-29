"""資料準備(Route A 取得側):HF 下載 → 扁平化 → 打 TAR → manifest。

來源:mater_kai_datascienctist.py
  - Route A(行 62–183):norm_rel / 分批 snapshot_download + rsync / 打 TAR / items.csv+paths.json
  - Step 3 扁平化(行 1193–1213):_normalize_rel / _ensure_hex2(XX/ 分桶)

這支是 hub.publish_dataset 的**上游**:產出 train_dir(扁平化的 ImageFolder)、單一 .tar、
items.csv/paths.json;再交給 hub.verify_dataset 健檢、hub.publish_dataset 版本化上傳。

重構重點:
  - norm_rel 在原始碼被重複定義 5 次 → 收斂成單一可信實作 normalize_rel。
  - 分批下載維持「可續傳、冪等」的韌性(resume_download + rsync --ignore-existing + 429 降速重試)。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from ..configs import CFG

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def normalize_rel(path: str) -> str:
    """正規化資料集內相對路徑:統一斜線、去重複斜線、剝掉 raw/train、train/train、train、./train 前綴。

    (原始碼裡這段邏輯被重複定義了 5 次 —— 收斂成單一實作。)
    """
    p = str(path).strip().replace("\\", "/")
    p = re.sub(r"/+", "/", p).lstrip("/")
    for head in ("raw/train/", "train/train/", "train/", "./train/"):
        if p.startswith(head):
            return p[len(head):]
    return p


# ── 小工具 ────────────────────────────────────────────────────────────────
def _hash_prefix(path: Path, algo: str = "md5", n: int = 2) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()


def _count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def _rsync(src: Path, dst: Path) -> None:
    """冪等鏡像:--ignore-existing,已存在的就不重抓(可重複跑、可續)。"""
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.call(["rsync", "-a", "--ignore-existing", f"{src}/", f"{dst}/"])


def _token(token: str | None) -> str | None:
    return token or CFG.hf_token or None


# ── 1) 分批下載(可續傳、冪等)──────────────────────────────────────────────
def download_images(repo_id: str, dest_dir, *, group_size: int = 8, max_workers: int = 8,
                    drive_dir=None, revision: str = "main", token: str | None = None) -> int:
    """把 00..ff 拆成每組 group_size 個前綴,分批從 HF 下載 train/** 到 dest_dir,回傳張數。

    韌性:resume_download 可續傳;429/失敗自動降速重試一次;每批可選 rsync 到 drive_dir(永久層)。
    """
    from huggingface_hub import snapshot_download
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    hex2 = [f"{i:02x}" for i in range(256)]
    groups = [hex2[i:i + group_size] for i in range(0, 256, group_size)]

    for gi, group in enumerate(groups, 1):
        patterns = [pat for px in group for pat in (f"train/train/{px}/*", f"train/{px}/*")]
        print(f"=== 批次 {gi}/{len(groups)}:{group[0]}..{group[-1]} ===")
        for workers in (max_workers, max(1, max_workers // 2)):  # 失敗就降速重試一次
            try:
                snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                  allow_patterns=patterns, local_dir=str(dest_dir),
                                  resume_download=True, max_workers=workers, token=_token(token))
                break
            except Exception as e:
                print(f"[warn] workers={workers} 失敗:{e}")
        if drive_dir:
            _rsync(dest_dir, Path(drive_dir))
    return _count_images(dest_dir)


# ── 2) 扁平化成 XX/filename ────────────────────────────────────────────────
def flatten_to_hex2(src_dir, dst_dir, *, dry_run: bool = False) -> dict:
    """把任意巢狀的影像扁平化成 `XX/filename` 兩層結構(XX = hex 分桶),回傳統計。

    分桶規則(對齊 Step 3):
      1. 已是 XX/filename → 保留 XX。
      2. 檔名前兩碼是十六進位(常見 hash 檔名)→ 用那兩碼。
      3. 否則回退用檔案 MD5 前兩碼(保底,確保一定有桶)。
    dry_run=True 只統計、不搬檔。
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    moved = skipped = 0
    for p in src_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _IMG_EXTS:
            continue
        parts = normalize_rel(p.relative_to(src_dir).as_posix()).split("/")
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{2}", parts[0]):
            bucket, fn = parts[0].lower(), "/".join(parts[1:])
        else:
            fn = p.name
            m = re.match(r"^([0-9a-fA-F]{2})", fn)
            bucket = m.group(1).lower() if m else _hash_prefix(p)
        if dry_run:
            moved += 1
            continue
        dst = dst_dir / bucket / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == p.stat().st_size:
            skipped += 1
        else:
            shutil.copy2(p, dst)
            moved += 1
    return {"moved": moved, "skipped": skipped, "dst_dir": str(dst_dir)}


# ── 3) manifest + TAR ─────────────────────────────────────────────────────
def build_manifest(train_dir, *, items_csv=None, paths_json=None) -> list[str]:
    """掃 train_dir,產出排序後的相對路徑清單,寫 items.csv + paths.json,回傳清單。"""
    train_dir = Path(train_dir)
    rels = sorted(p.relative_to(train_dir).as_posix()
                  for p in train_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    if items_csv:
        with Path(items_csv).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path"])
            w.writerows([[r] for r in rels])
    if paths_json:
        Path(paths_json).write_text(json.dumps(rels, ensure_ascii=False), encoding="utf-8")
    return rels


def pack_tar(train_dir, tar_path) -> dict:
    """把 train_dir(ImageFolder)打成單一未壓縮 .tar,回傳 {path, sha256, size_bytes}。

    註:這是「單一 ImageFolder tar」(對齊原專案);WDS 讀它時每個 sample 只有影像,
    caption 由 cap_map.jsonl 外掛覆寫(見 captions.py)。要做正規 WebDataset 分片(多 tar、
    每 sample 配對 image+caption)是可選改進。
    """
    train_dir, tar_path = Path(train_dir), Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:        # "w" = 不壓縮,最快
        tar.add(train_dir, arcname=train_dir.name)
    return {"path": str(tar_path),
            "sha256": _hash_prefix(tar_path, "sha256", n=0),
            "size_bytes": tar_path.stat().st_size}

# -*- coding: utf-8 -*-
"""AI Fashion — Colab 一鍵真跑(self-contained,不需安裝 aifashion 套件)。

整理自原始 V7 notebook「實際執行」的 Cell 2–7(Cell 6.5A-reset 訓練 + Cell 7 評估),
清乾淨、指向你 Drive 的 shards,可直接貼進 Colab 一個 cell 執行。

用法:
  1) Colab 執行階段 → 變更類型 → GPU(T4)。
  2) 把整個檔案貼進一個 cell 執行(或上傳後 %run colab_run.py)。
  3) 視需要改 CONFIG 的 SRC_DIR。
  4) 預設只跑 zero-shot 評估(快、馬上看到真 R@K);要訓練把 DO_TRAIN=True。

只裝 Colab 缺的兩個套件(webdataset / open_clip),不碰 datasets/faiss/gradio,
避免相依爆炸把 pip 拖慢。
"""

# ========== 0) 安裝(只補 Colab 缺的)==========
import importlib, subprocess, sys


def _ensure(mod, pip_name=None):
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"安裝 {pip_name or mod} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)


_ensure("webdataset")
_ensure("open_clip", "open_clip_torch")

# ========== 1) CONFIG(改這裡)==========
SRC_DIR = "/content/drive/MyDrive/第二組應用班/shards"      # ← 你的 39 個 .tar 所在資料夾
RUN_DIR = "/content/drive/MyDrive/kai_outfit_runs/clean_v7"  # 輸出/讀 cap_map 的目錄
MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"
BATCH, EVAL_MAX_N = 32, 1500

DO_TRAIN = False            # True = 跑 anchor-to-T0 訓練,再比較 finetuned
TRAIN_STEPS, LR, ACCUM = 1200, 5e-4, 2

# ========== 2) imports + mount + device ==========
import glob
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

import open_clip
import webdataset as wds

try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass

Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ========== 3) robust WDS:讀圖 + caption 解析鏈 + cap_map 覆寫 + 噪聲過濾 ==========
IMG_KEYS = ["jpg", "jpeg", "png", "webp", "bmp"]
TXT_KEYS = ["txt", "caption", "caption.txt", "json"]


def _clean_words(s):
    s = os.path.splitext(os.path.basename(str(s)))[0]
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _heuristic_caption(sample):
    folder = os.path.basename(os.path.dirname(sample.get("__url__", "")))
    parts = []
    if folder and folder.lower() not in {"train", "val", "validation", "test"}:
        parts.append(_clean_words(folder))
    if sample.get("__key__"):
        parts.append(_clean_words(sample["__key__"]))
    parts.append("outfit")
    return " ".join(p for p in parts if p)


_CAP_MAP = None


def _load_cap_map():
    mp, p = {}, Path(RUN_DIR) / "cap_map.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                k = o.get("key") or o.get("image_path") or o.get("__key__")
                v = o.get("caption")
                if k and v:
                    mp[str(k)] = str(v)
            except Exception:
                pass
    print(f"cap_map 載入 {len(mp)} 筆" + ("" if mp else "(沒有 cap_map.jsonl → 用啟發式 caption)"))
    return mp


def _extract(sample):
    global _CAP_MAP
    img = None
    for k in IMG_KEYS:
        if k in sample:
            v = sample[k]
            try:
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

    cap = None
    js = sample.get("json")
    if isinstance(js, dict):
        cap = js.get("caption") or js.get("text")
    if cap is None:
        for k in TXT_KEYS:
            if k in sample and k != "json":
                v = sample[k]
                cap = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                break
    if not cap or not cap.strip():
        cap = _heuristic_caption(sample)

    if _CAP_MAP is None:
        _CAP_MAP = _load_cap_map()
    key = sample.get("__key__", "")
    if key in _CAP_MAP:
        cap = _CAP_MAP[key]

    if len(re.findall(r"[a-z0-9]+", str(cap).lower())) < 3:   # 噪聲 caption 丟棄
        return None
    return {"image": img, "caption": cap, "key": key}


class _Gen(IterableDataset):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        yield from self.fn()


def make_stream(urls):
    ds = wds.WebDataset(urls, resampled=False).map(_extract).select(lambda x: x is not None)

    def gen():
        for x in ds:
            yield x["image"], x["caption"], x["key"]
    return gen


def make_loader(stream, preprocess, tokenizer, bs=BATCH):
    def collate(batch):
        imgs, caps, keys = zip(*batch)
        return torch.stack([preprocess(im) for im in imgs], 0), tokenizer(list(caps)), list(caps), list(keys)
    return DataLoader(_Gen(stream), batch_size=bs, num_workers=0, collate_fn=collate, pin_memory=True)


# ========== 4) 找 shards ==========
urls = sorted(glob.glob(os.path.join(SRC_DIR, "*.tar")))
assert urls, f"❌ 找不到任何 .tar:{SRC_DIR}(改 CONFIG 的 SRC_DIR)"
print(f"找到 {len(urls)} 個 shard")

# ========== 5) 模型(凍結 backbone)+ anchor-to-T0 投影頭 ==========
model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=device)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
for p in model.parameters():
    p.requires_grad = False
model.eval()
DIM = int(getattr(model.visual, "output_dim", 512))


@torch.no_grad()
def encode_images(x):
    return F.normalize(model.encode_image(x.to(device)), dim=-1)


@torch.no_grad()
def encode_text(t):
    return F.normalize(model.encode_text(t.to(device)), dim=-1)


img_proj = nn.Linear(DIM, DIM, bias=False).to(device)
txt_proj = nn.Linear(DIM, DIM, bias=False).to(device)
logit_scale = nn.Parameter(torch.tensor(0.07).log().to(device))


def _eye(linear):
    with torch.no_grad():
        linear.weight.zero_()
        n = min(linear.weight.shape)
        linear.weight[:n, :n] = torch.eye(n, device=device)


_eye(img_proj)
_eye(txt_proj)
for p in txt_proj.parameters():       # 文字端錨定:凍結成 Identity
    p.requires_grad = False

# 若 RUN_DIR 有先前訓練好的 last_extra.pt,載回來
_ckpt = Path(RUN_DIR) / "last_extra.pt"
if _ckpt.exists() and not DO_TRAIN:
    sd = torch.load(_ckpt, map_location=device)
    img_proj.load_state_dict(sd["img_proj"])
    print("已載入先前訓練的 img_proj:", _ckpt)

# ========== 6) R@K + 收集 + 報告 ==========
def r_at_k(S, k):
    ranks = np.argsort(-S, axis=1)[:, :k]
    gt = np.arange(S.shape[0])[:, None]
    return float((ranks == gt).any(axis=1).mean())


@torch.no_grad()
def collect(loader, max_n=EVAL_MAX_N):
    I, T, n = [], [], 0
    for imgs, toks, _, _ in tqdm(loader, desc="collect features"):
        I.append(encode_images(imgs).cpu())
        T.append(encode_text(toks).cpu())
        n += imgs.size(0)
        if n >= max_n:
            break
    I, T = torch.cat(I, 0), torch.cat(T, 0)
    N = min(I.shape[0], T.shape[0], max_n)
    return I[:N], T[:N]


@torch.no_grad()
def report(tag, I0, T0):
    I1 = F.normalize(img_proj(I0.to(device)), dim=-1).cpu()
    S0 = (T0 @ I0.t()).numpy()           # zero-shot:T0 @ I0ᵀ
    S1 = (T0 @ I1.t()).numpy()           # finetuned:T0 @ I1ᵀ
    print(f"[{tag}] zero-shot  R@1 {r_at_k(S0,1):.3f}  R@5 {r_at_k(S0,5):.3f}  R@10 {r_at_k(S0,10):.3f}")
    print(f"[{tag}] finetuned  R@1 {r_at_k(S1,1):.3f}  R@5 {r_at_k(S1,5):.3f}  R@10 {r_at_k(S1,10):.3f}")


# ========== 7) 評估(一定跑)==========
val_loader = make_loader(make_stream(urls), preprocess_eval, tokenizer)
I0, T0 = collect(val_loader)
print(f"\n收集到 {I0.shape[0]} 對 (image, text)")
report("Eval", I0, T0)

# ========== 8) 可選:anchor-to-T0 訓練(只訓影像頭)==========
if DO_TRAIN:
    print("\n=== 開始訓練 anchor-to-T0(只訓 img_proj)===")
    train_loader = make_loader(make_stream(urls), preprocess_train, tokenizer)
    opt = torch.optim.AdamW(img_proj.parameters(), lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    img_proj.train()
    opt.zero_grad(set_to_none=True)
    step = 0
    pbar = tqdm(total=TRAIN_STEPS, desc="train anchor-to-T0")
    while step < TRAIN_STEPS:
        for imgs, toks, _, _ in train_loader:
            fi, t0 = encode_images(imgs), encode_text(toks)        # 凍結 backbone 向量
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                zi = F.normalize(img_proj(fi.to(device)), dim=-1)
                scale = logit_scale.exp().clamp(1e-3, 100)
                sim = scale * (zi @ t0.to(device).t())             # 單向 in-batch 對比
                loss = F.cross_entropy(sim, torch.arange(sim.size(0), device=device)) / ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM == 0:
                scaler.unscale_(opt)                               # AMP 下先 unscale 再 clip
                torch.nn.utils.clip_grad_norm_(img_proj.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.item() * ACCUM))
            if step >= TRAIN_STEPS:
                break
    pbar.close()
    img_proj.eval()
    torch.save({"img_proj": img_proj.state_dict(), "txt_proj": txt_proj.state_dict(),
                "logit_scale": logit_scale.detach().float().cpu()}, _ckpt)
    print("已存:", _ckpt)
    print("\n=== 訓練後重評(finetuned 那列應 ≥ 訓練前)===")
    report("After-train", I0, T0)

print("\n✅ 完成")

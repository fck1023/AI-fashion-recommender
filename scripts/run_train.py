"""一鍵訓練影像投影頭(對齊文字 anchor T0)。

    python scripts/run_train.py --prefer hf --steps 200
    python scripts/run_train.py --cap-map runs/v7/cap_map.jsonl   # 套用 BLIP 強化 caption

超參不給就取 configs.TrainConfig 預設。需先 `pip install -e .`。
"""
from __future__ import annotations

import argparse

from aifashion.configs import CFG
from aifashion.data.captions import load_cap_map
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.model import AnchoredCLIP
from aifashion.train import train_image_head


def main() -> None:
    ap = argparse.ArgumentParser(description="訓練影像投影頭(anchor-to-T0)")
    ap.add_argument("--prefer", choices=["drive", "hf"], default="drive", help="資料來源(drive 找不到自動回退 hf)")
    ap.add_argument("--shards-dir", default=None, help="顯式 .tar shards 目錄(如 Drive 路徑)")
    ap.add_argument("--cap-map", default=None, help="cap_map.jsonl 路徑(套用 BLIP 強化 caption)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--copy-ssd", action="store_true", help="shards 先 rsync 到本機 SSD(Colab 較快)")
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = AnchoredCLIP.from_pretrained(CFG.model.name, CFG.model.pretrained, device=args.device)
    cap_map = load_cap_map(args.cap_map) if args.cap_map else None
    train_stream, val_stream, source = build_streams(
        prefer=args.prefer, shards_dir=args.shards_dir, copy_ssd=args.copy_ssd, cap_map=cap_map)
    print(f"資料來源:{source}")

    train_loader = make_loader(train_stream, model.preprocess_train, model.tokenizer, args.batch_size)
    val_loader = make_loader(val_stream, model.preprocess_eval, model.tokenizer, args.batch_size)

    save_path = train_image_head(model, train_loader, val_loader,
                                 lr=args.lr, steps=args.steps, grad_accum=args.grad_accum,
                                 save_path=args.save_path)
    print(f"✅ 訓練完成,影像頭存於:{save_path}")


if __name__ == "__main__":
    main()

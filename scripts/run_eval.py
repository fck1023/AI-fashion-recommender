"""一鍵評估:zero-shot vs finetuned 的 R@K → 存 compare.json。

    python scripts/run_eval.py                          # 用 Identity 起點(等同 zero-shot 對照)
    python scripts/run_eval.py --heads runs/v7/last_extra.pt   # 載入訓練好的投影頭

需先 `pip install -e .`。
"""
from __future__ import annotations

import argparse

from aifashion.configs import CFG
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.eval import evaluate
from aifashion.model import AnchoredCLIP


def main() -> None:
    ap = argparse.ArgumentParser(description="評估 R@K(zero-shot vs finetuned)")
    ap.add_argument("--heads", default=None, help="訓練好的投影頭 .pt(不給=Identity 起點)")
    ap.add_argument("--prefer", choices=["drive", "hf"], default="drive")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-dir", default=None, help="compare.json 輸出目錄(預設 configs.run_dir)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = AnchoredCLIP.from_pretrained(CFG.model.name, CFG.model.pretrained, device=args.device)
    if args.heads:
        model.load_heads(args.heads)
        print(f"已載入投影頭:{args.heads}")

    _, val_stream, source = build_streams(prefer=args.prefer)
    print(f"資料來源:{source}")
    val_loader = make_loader(val_stream, model.preprocess_eval, model.tokenizer, args.batch_size)

    metrics = evaluate(model, val_loader, max_samples=args.max_samples, run_dir=args.run_dir)
    print(metrics)


if __name__ == "__main__":
    main()

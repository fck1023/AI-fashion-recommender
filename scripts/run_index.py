"""一鍵建索引:抽全庫影像向量 → FAISS/HNSW → 存 4 個 artifacts(可選發佈 HF)。

    python scripts/run_index.py --max-n 2000
    python scripts/run_index.py --heads runs/v7/last_extra.pt --project \\
        --publish Kai1023/kai-outfit-ai-dataset

預設用 zero-shot backbone 向量(ablation 顯示檢索最佳);--project 改用訓練後投影 I1。
需先 `pip install -e .`。
"""
from __future__ import annotations

import argparse

from aifashion.configs import CFG
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.index import encode_images_to_vectors, save_artifacts
from aifashion.model import AnchoredCLIP


def main() -> None:
    ap = argparse.ArgumentParser(description="建近鄰索引 + 存/發佈 artifacts")
    ap.add_argument("--heads", default=None, help="訓練好的投影頭 .pt(搭配 --project 使用)")
    ap.add_argument("--project", action="store_true", help="用訓練後投影 I1(預設用 zero-shot I0)")
    ap.add_argument("--prefer", choices=["drive", "hf"], default="drive")
    ap.add_argument("--max-n", type=int, default=0, help="只抽前 N 張(0=全部)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--publish", default=None, help="HF repo_id;給了就發佈 + 更新 latest")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = AnchoredCLIP.from_pretrained(CFG.model.name, CFG.model.pretrained, device=args.device)
    if args.heads:
        model.load_heads(args.heads)
        print(f"已載入投影頭:{args.heads}")

    gallery_stream, _, source = build_streams(prefer=args.prefer)
    print(f"資料來源:{source}")
    loader = make_loader(gallery_stream, model.preprocess_eval, model.tokenizer, args.batch_size)

    vecs, keys = encode_images_to_vectors(model, loader, project=args.project, max_n=args.max_n)
    print(f"抽到向量:{vecs.shape}(project={args.project})")
    run_dir = save_artifacts(vecs, keys, run_dir=args.run_dir)

    if args.publish:
        from aifashion.data.hub import publish_artifacts
        info = publish_artifacts(run_dir, args.publish)
        print(f"✅ 已發佈:{info}")


if __name__ == "__main__":
    main()

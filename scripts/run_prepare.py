"""一鍵資料準備(Route A):下載 → 扁平化 → manifest → 打 TAR → 驗證 →(可選)發佈 HF。

    python scripts/run_prepare.py --repo-id Kai1023/kai-outfit-ai-dataset \\
        --work-dir /content/ds \\
        --train-dir /content/drive/MyDrive/OutfitData/raw/train --publish

下載/Drive 寫入屬重操作,通常在 Colab + Drive 環境跑。需先 `pip install -e .`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aifashion.data import hub, prepare


def main() -> None:
    ap = argparse.ArgumentParser(description="Route A 資料準備 + 雙雲發佈")
    ap.add_argument("--repo-id", required=True, help="HF dataset repo")
    ap.add_argument("--work-dir", default="./_work", help="下載暫存目錄")
    ap.add_argument("--train-dir", default="./_work/train", help="扁平化後的 ImageFolder(永久層,可放 Drive)")
    ap.add_argument("--group-size", type=int, default=8, help="每批下載的 hex 前綴數")
    ap.add_argument("--skip-download", action="store_true", help="已有 raw 影像時跳過下載")
    ap.add_argument("--publish", action="store_true", help="驗證後發佈到 HF Hub")
    args = ap.parse_args()

    work, train_dir = Path(args.work_dir), Path(args.train_dir)

    if not args.skip_download:
        n = prepare.download_images(args.repo_id, work, group_size=args.group_size)
        print(f"下載完成,raw 影像數:{n}")

    print(f"扁平化:{prepare.flatten_to_hex2(work, train_dir)}")

    items = train_dir.parent / "items.csv"
    paths = train_dir.parent / "paths.json"
    rels = prepare.build_manifest(train_dir, items_csv=items, paths_json=paths)
    tar = prepare.pack_tar(train_dir, train_dir.parent / "train.tar")
    print(f"manifest {len(rels)} 筆;TAR sha256[:12]={tar['sha256'][:12]} size={tar['size_bytes']}")

    report = hub.verify_dataset(train_dir)
    report_path = train_dir.parent / "verify_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"驗證:{report}")

    if args.publish:
        oid = hub.publish_dataset(args.repo_id, tar_path=tar["path"], items_csv=items,
                                  paths_json=paths, verify_report=report_path)
        print(f"✅ 已發佈 commit:{oid}")


if __name__ == "__main__":
    main()

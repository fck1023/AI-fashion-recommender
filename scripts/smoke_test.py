"""Dry-run smoke test:不需資料、不需 GPU,驗證重構後的套件能 import 並跑通核心邏輯。

用途:在 Colab(或本機)裝好套件後,一行確認「整包沒壞」,再投入完整訓練。
    pip install -e .
    python scripts/smoke_test.py            # 基本(import + 純函式 + 合成索引)
    python scripts/smoke_test.py --model    # 額外載 CLIP backbone(會下載權重,CPU 即可)

每個階段獨立 try/except、個別回報 PASS/FAIL,最後輸出總結與離開碼(全過=0)。
"""
from __future__ import annotations

import argparse
import sys
import traceback

_results: list[tuple[str, bool]] = []


def _stage(name: str, fn) -> None:
    try:
        fn()
        _results.append((name, True))
        print(f"✅ PASS  {name}")
    except Exception as e:  # noqa: BLE001 — smoke test 要把任何失敗都收住、繼續跑下一階段
        _results.append((name, False))
        print(f"❌ FAIL  {name}\n     {e}")
        traceback.print_exc()


def s1_imports() -> None:
    """所有模組可 import + 設定載入 → 驗證套件結構與相依都接上。"""
    import aifashion.configs, aifashion.eval, aifashion.index, aifashion.model  # noqa: F401
    import aifashion.rerank, aifashion.train  # noqa: F401
    import aifashion.data.captions, aifashion.data.hub  # noqa: F401
    import aifashion.data.prepare, aifashion.data.webdataset  # noqa: F401
    from aifashion.configs import CFG
    assert CFG.model.name == "ViT-B-32"


def s2_pure_functions() -> None:
    """純函式:caption 覆寫鏈 / 噪聲過濾 / normalize_rel 收斂。"""
    from aifashion.data.captions import resolve_caption
    from aifashion.data.prepare import normalize_rel

    sample = {"__url__": "/x/train/3f", "__key__": "fa8c59_red_dress"}
    assert resolve_caption(sample, {"fa8c59_red_dress": "a red sleeveless cocktail dress"}) \
        == "a red sleeveless cocktail dress"                       # cap_map 覆寫
    assert resolve_caption({"json": {"caption": "red dress"}}, min_tokens=3) is None  # 噪聲丟棄
    assert normalize_rel("raw/train/3f/a.jpg") == "3f/a.jpg"       # 前綴剝除


def s3_recall_and_index() -> None:
    """R@K 算法 + 合成 FAISS 索引 roundtrip(用每個向量查自己,Top-1 必為自己)。"""
    import numpy as np
    from aifashion.eval import recall_at_k
    from aifashion.index import build_index, search

    assert recall_at_k(np.eye(20, dtype="float32"), 1) == 1.0      # 完美對角線 → R@1=1
    rng = np.random.default_rng(0)
    v = rng.standard_normal((100, 512)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids, _ = search(build_index(v, "faiss"), v[7], k=5)
    assert int(ids[0][0]) == 7


def s4_model() -> None:
    """載 AnchoredCLIP(CPU)+ 前向;驗證 Identity 起點 → 訓練前 I1 ≈ I0。"""
    import torch
    from aifashion.model import AnchoredCLIP

    m = AnchoredCLIP.from_pretrained(device="cpu")
    x = torch.randn(2, 3, 224, 224)
    i0 = m.encode_image_features(x)
    i1 = m.project_image(i0)
    assert i0.shape == i1.shape and i0.shape[0] == 2
    assert torch.allclose(i0, i1, atol=1e-4), "Identity-init 影像頭應使 I1 ≈ I0"


def s5_train_loop() -> None:
    """合成 mini loader 跑幾步訓練迴圈 + R@K 報告(驗證 loss/AMP/存檔路徑全通,純 CPU)。"""
    import pathlib
    import tempfile

    import torch
    from aifashion.model import AnchoredCLIP
    from aifashion.train import train_image_head

    m = AnchoredCLIP.from_pretrained(device="cpu")
    tokens = m.tokenizer(["a red dress"] * 8)

    def fake_loader():  # 產出 (images, tokens, captions, keys) contract
        return [(torch.randn(8, 3, 224, 224), tokens, ["a red dress"] * 8,
                 [f"k{i}" for i in range(8)]) for _ in range(3)]

    save_path = pathlib.Path(tempfile.mkdtemp()) / "heads.pt"
    train_image_head(m, fake_loader(), fake_loader(),
                     steps=6, grad_accum=2, amp=False, save_path=save_path)
    assert save_path.exists(), "train_image_head 應存出影像頭"


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Fashion 套件 dry-run smoke test")
    ap.add_argument("--model", action="store_true", help="額外載 CLIP backbone(下載權重,CPU 即可)")
    ap.add_argument("--train", action="store_true", help="額外跑合成資料的迷你訓練迴圈(含 R@K 報告)")
    args = ap.parse_args()

    _stage("1) 套件 import + 設定", s1_imports)
    _stage("2) 純函式(caption 鏈 / normalize_rel)", s2_pure_functions)
    _stage("3) R@K + 合成 FAISS 索引 roundtrip", s3_recall_and_index)
    if args.model:
        _stage("4) 載 AnchoredCLIP + 前向 + Identity 起點", s4_model)
    if args.train:
        _stage("5) 迷你訓練迴圈 + R@K 報告(合成資料)", s5_train_loop)

    ok = sum(1 for _, passed in _results if passed)
    print(f"\n=== Smoke test:{ok}/{len(_results)} 階段通過 ===")
    sys.exit(0 if ok == len(_results) else 1)


if __name__ == "__main__":
    main()

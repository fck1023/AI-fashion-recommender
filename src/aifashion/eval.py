"""評估:R@K 檢索指標(zero-shot vs finetuned)。

來源:ai_fashion_..._blip_v7.py Cell 7(collect_backbone_features / r_at_k_numpy /
     報告 I0–T0 vs I1–T0)+ Cell 6.5A-reset 的 quick eval(_report_pre_post)。

──────────────────────────────────────────────────────────────────────────
R@K 是什麼、為什麼用它(面試講得出來)
──────────────────────────────────────────────────────────────────────────
檢索任務:給一段文字 query,從一堆候選影像裡找「對的那張」。把驗證集 N 對
(文字 Tᵢ ↔ 影像 Iᵢ)排成一個 N×N 相似度矩陣 S,**第 i 列的對角元素 S[i,i]
就是正確配對**。R@K = 「每列前 K 名裡有命中對角線」的比例。

  - R@1  嚴格:最高分必須剛好是對的那張。
  - R@10 寬鬆:前 10 名有對的就算數(本專案主指標,V7 目標 ~0.7)。

關鍵性質:**R@K 只看排序,不看分數大小**。所以把整個 S 乘上任何正常數
(例如溫度 logit_scale)排序不變、R@K 不變 —— 原 notebook Cell 7 只對
finetuned 乘了 scale、zero-shot 沒乘,那行其實無害但會誤導,這裡一律不乘。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .configs import CFG


def recall_at_k(similarity: np.ndarray, k: int) -> float:
    """給相似度矩陣(query × gallery,對角線為正解),算 Recall@K。

    對每一列取分數最高的前 K 個 gallery index;只要其中有「該列自己的 index」
    (= 對角線正解)就算命中。回傳所有列的命中率。
    """
    sim = np.asarray(similarity)
    topk_idx = np.argsort(-sim, axis=1)[:, :k]          # 每列分數由高到低取前 K 個欄位
    ground_truth = np.arange(sim.shape[0])[:, None]     # 第 i 列的正解就是第 i 欄
    return float((topk_idx == ground_truth).any(axis=1).mean())


@torch.no_grad()
def collect_image_text_features(model, loader, max_n: int):
    """掃驗證集,收集凍結 backbone 的影像向量 I0 與文字 anchor T0(都已 L2 normalize)。

    loader 每個 batch 產出 (images, tokens, captions, keys);這裡只用前兩個。
    收滿 max_n 對就停,回傳對齊長度的 (I0, T0),都在 CPU 上方便算矩陣。
    """
    img_feats, txt_feats, n = [], [], 0
    for batch in loader:
        images, tokens = batch[0], batch[1]
        i0 = model.encode_image_features(images)        # 凍結 backbone 影像向量
        t0 = model.encode_text_anchor(tokens)           # 文字 anchor(txt_proj 凍結為 Identity)
        img_feats.append(i0.cpu())
        txt_feats.append(t0.cpu())
        n += i0.size(0)
        if n >= max_n:
            break
    I0 = torch.cat(img_feats, 0)
    T0 = torch.cat(txt_feats, 0)
    N = min(I0.shape[0], T0.shape[0], max_n)
    return I0[:N], T0[:N]


def _print_table(metrics: dict, tag: str, ks) -> None:
    """把 zero-shot vs finetuned 印成對齊的小表(不依賴 pandas)。"""
    header = f"[{tag}] " if tag else ""
    cols = [f"R@{k}" for k in ks]
    print(f"\n{header}{'':12s}" + "".join(f"{c:>10s}" for c in cols) + f"{'n':>10s}")
    for name in ("zero", "finetuned"):
        row = metrics[name]
        vals = "".join(f"{row[c]:>10.3f}" for c in cols)
        print(f"{header}{name:12s}{vals}{row['n']:>10d}")


def report_zero_shot_vs_finetuned(model, loader, max_n: int = 1500,
                                  tag: str = "", ks=(1, 5, 10)) -> dict:
    """同一批驗證資料,比較「不訓練」與「訓練後影像頭」的 R@K。

    - zero-shot   :相似度 = T0 @ I0ᵀ(完全沒動投影頭,純 backbone)。
    - finetuned   :相似度 = T0 @ I1ᵀ,I1 = 過了可訓練 img_proj 的影像向量。
    兩者用「同一組 T0、同一批樣本」比較,差異才完全來自影像頭。

    訓練前呼叫:兩列應該幾乎相等(Identity 起點,I1 ≈ I0)→ 確認沒從壞權重出發。
    訓練後呼叫:finetuned 那列理想上 ≥ zero-shot,代表影像頭真的把 I1 推向 anchor。
    """
    I0, T0 = collect_image_text_features(model, loader, max_n)
    I1 = model.project_image(I0).cpu()                  # 訓練中的影像向量

    # 相似度矩陣:列 = 文字 query,欄 = 影像 gallery,對角線為正解。
    # 不乘 logit_scale —— R@K 只看排序,乘正常數不影響結果(見模組 docstring)。
    sim_zero = (T0 @ I0.t()).numpy()
    sim_finetuned = (T0 @ I1.t()).numpy()

    n = int(I0.shape[0])
    metrics = {
        "zero": {f"R@{k}": recall_at_k(sim_zero, k) for k in ks},
        "finetuned": {f"R@{k}": recall_at_k(sim_finetuned, k) for k in ks},
    }
    metrics["zero"]["n"] = metrics["finetuned"]["n"] = n
    _print_table(metrics, tag, ks)
    return metrics


def evaluate(model, val_loader, *, max_samples: int | None = None,
             run_dir=None, ks=(1, 5, 10)) -> dict:
    """跑完整評估並把比較表存成 compare.json,回傳 metrics(對應 Cell 7)。"""
    max_samples = CFG.eval.max_samples if max_samples is None else max_samples
    metrics = report_zero_shot_vs_finetuned(model, val_loader, max_samples, tag="Eval", ks=ks)

    run_dir = Path(run_dir or CFG.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "compare.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 已存比較表:{run_dir / 'compare.json'}")
    return metrics

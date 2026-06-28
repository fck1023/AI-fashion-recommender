"""評估:R@K 檢索(zero-shot vs finetuned)。

來源:ai_fashion_..._blip_v7.py Cell 7(collect_backbone_features /
     r_at_k_numpy / 報告 I0–T0 vs I1–T0)。

R@K = 前 K 名命中正確配對的比例;本專案目標 R@10 > ~0.7(V7 達 0.706)。
會輸出 compare.json / compare_table.csv 比較 zero-shot 與 finetuned。
"""
from __future__ import annotations

import numpy as np


def recall_at_k(similarity: np.ndarray, k: int) -> float:
    """給相似度矩陣(query×gallery),算 R@K。"""
    raise NotImplementedError("Phase 1:自 v7 Cell 7 的 r_at_k_numpy 搬入")


def evaluate(max_samples: int = 1500, topk: int = 10) -> dict:
    """跑 zero-shot vs finetuned 的 R@1/5/10,回傳比較表。"""
    raise NotImplementedError("Phase 1")

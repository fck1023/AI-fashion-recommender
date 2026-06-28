"""近鄰索引:把影像向量建成可快速檢索的索引。

來源:ai_fashion_..._blip_v7.py Cell 9(建索引)/ mater_kai(FAISS IP)/
     處理資料集part2.py(hnswlib)。

設計重點:
  - 推薦 = 在影像向量索引裡找「與文字向量最近」的 Top-K(內積/cosine)。
  - 兩種後端:FAISS(快、需安裝)與 hnswlib(免 faiss、記憶體友善)擇一。
  - 同時存出 UI 需要的檔案(keys.npy / meta.json / 縮圖)。
"""
from __future__ import annotations

import numpy as np


def build_index(vectors: np.ndarray, backend: str = "faiss"):
    """以影像向量建立 ANN 索引(faiss / hnswlib),回傳索引物件。"""
    raise NotImplementedError("Phase 1:整併 v7 Cell 9 / mater_kai / part2 的重複實作")


def search(index, query_vec: np.ndarray, k: int = 10):
    """查 Top-K,回傳 (indices, scores)。"""
    raise NotImplementedError("Phase 1")

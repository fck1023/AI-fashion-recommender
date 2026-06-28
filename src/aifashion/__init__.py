"""AI Fashion Recommender — 以 CLIP 做「文字 → 穿搭」跨模態檢索。

公開介面在各子模組:
  - model   : CLIP backbone + 影像投影頭(anchor-to-T0)
  - train   : 訓練影像頭對齊文字 anchor
  - eval    : R@K 檢索評估(zero-shot vs finetuned)
  - index   : 建近鄰索引(FAISS / HNSW)
  - rerank  : 屬性 / 知識圖譜 / BLIP-ITM 重排序
  - data.*  : WebDataset 讀取與 caption 生成
"""
__version__ = "0.1.0"

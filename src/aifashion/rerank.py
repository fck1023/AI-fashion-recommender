"""重排序:先粗排(向量檢索)再精排,提升 Top-K 品質。

來源:ai_fashion_..._blip_v7.py Cell 9(屬性零樣本打分 + 迷你知識圖譜 KG 重排序)
     與 Cell 9R(BLIP-ITM 交叉編碼精排)。

設計重點(典型「retrieval → re-rank」兩段式):
  - 第一段:用向量索引快速取回候選(preselect ~60)。
  - 第二段:用較貴但較準的訊號重排:
      * 屬性零樣本打分(袖長/外套…)+ 迷你 KG 規則。
      * BLIP-ITM 交叉編碼(圖文是否匹配)做最終精排。
"""
from __future__ import annotations


def attribute_score(image, text: str) -> float:
    """零樣本屬性打分(袖長/外套/領型…)。"""
    raise NotImplementedError("Phase 1:自 v7 Cell 9 搬入")


def blip_itm_rerank(images: list, text: str, topk: int) -> list:
    """用 BLIP-ITM 交叉編碼對候選做精排,回傳重排後順序。"""
    raise NotImplementedError("Phase 1:自 v7 Cell 9R 搬入")

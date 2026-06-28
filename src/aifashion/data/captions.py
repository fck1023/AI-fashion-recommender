"""Caption 生成:讓文字監督訊號「可學」。

來源:ai_fashion_..._blip_v7.py Cell 3(_clean_words / _heuristic_caption /
     _load_cap_map / _extract)+ BLIP captioning。

設計重點(面試可講):
  V5/V6 失敗的主因之一是 caption 太弱(短、重複、季節詞而非服飾細節)。
  V7 改用 BLIP 生成句子 + 服飾屬性短詞(袖長/外套/領型/褲型…),讓文字描述
  更貼近影像實體,文字監督才「學得動」。
"""
from __future__ import annotations


def heuristic_caption(sample: dict) -> str:
    """從檔名/資料夾推一個保底 caption(無人工標註時用)。"""
    raise NotImplementedError("Phase 1：自 v7 Cell 3 搬入")


def blip_caption(image) -> str:
    """用 BLIP 生成影像描述(V7 的關鍵強化)。"""
    raise NotImplementedError("Phase 1")


def load_cap_map(path: str) -> dict:
    """載入人工/半自動標註的 caption 覆寫表(cap_map.jsonl)。"""
    raise NotImplementedError("Phase 1")

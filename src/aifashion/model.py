"""模型:CLIP backbone + 影像投影頭(anchor-to-T0)。

來源:ai_fashion_..._blip_v7.py Cell 5(open_clip 載入、encode_images /
     encode_text)+ Cell 6.5(_eye_init:投影頭以 Identity 初始化)。

設計重點(面試第一個主打 — space drift 的修法):
  - 文字端固定為「anchor」(txt_proj = Identity 且 freeze),不訓練。
  - 只訓練「影像投影頭」I1 去對齊文字 anchor T0,避免影像/文字兩邊一起動
    造成的 embedding 空間漂移(space drift)。
  - 從壞權重起手時先 reset 成 Identity,確保訓前 I1–T0 ≈ I0–T0。
"""
from __future__ import annotations

import torch.nn as nn


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    """載入 open_clip backbone、preprocess 與 tokenizer。"""
    raise NotImplementedError("Phase 1：自 v7 Cell 5 搬入")


class ImageProjectionHead(nn.Module):
    """影像投影頭;以 Identity 初始化,只訓練這一塊去對齊文字 anchor。"""

    def __init__(self, dim: int):
        super().__init__()
        raise NotImplementedError("Phase 1：自 v7 Cell 6.5 的 _eye_init 搬入")

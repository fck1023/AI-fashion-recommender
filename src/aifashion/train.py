"""訓練:只訓影像投影頭,對齊文字 anchor T0。

來源:ai_fashion_..._blip_v7.py Cell 6.5A(_enc / _loss_anchor_T0 / 訓練迴圈 /
     存 last.pt)。

設計重點:
  - Loss 只把影像向量 I1 拉近對應的文字 anchor T0(對比式)。
  - 訓前先做 quick eval:I1–T0 應 ≈ I0–T0(確認沒從壞起點出發)。
  - 超參集中在 configs.TrainConfig(batch/accum/lr/epochs/steps/amp/seed)。
"""
from __future__ import annotations


def train_image_head() -> str:
    """訓練影像投影頭並存檔(last.pt),回傳權重路徑。"""
    raise NotImplementedError("Phase 1：自 v7 Cell 6.5A 搬入(去除 Colab 專用碼)")


if __name__ == "__main__":
    train_image_head()

"""Gradio 互動式 demo:文字查詢 → Top-K 穿搭推薦。

來源:ai_fashion_..._blip_v7.py Cell 9 的新 UI / 驗證一下master_kai_的成果.py 的
     Gradio 介面(三種查詢模式)。

功能:
  - 文字查詢 + 多維度過濾(風格/單品/季節…)。
  - 顯示 Top-K 縮圖 + cosine 相似度。
  - 可選 KG / BLIP-ITM 重排序。
  - 從 HF Hub 的 `latest` 別名載入 artifacts(index/keys/meta),重開也能讀。
"""
from __future__ import annotations


def build_demo():
    """組裝並回傳 Gradio app(尚未啟動)。"""
    raise NotImplementedError("Phase 1:自 v7 Cell 9 / 驗證檔的 UI 搬入並去除 Colab 專用碼")


if __name__ == "__main__":
    build_demo().launch()

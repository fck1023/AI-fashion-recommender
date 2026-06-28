# notebooks/

放原始 Colab notebook(需 GPU / Google Drive,僅供重現與遷移對照)。

**遷移來源(Phase 1 重構的依據)** — 原始碼在:
`/Users/kai/Desktop/Kai/123/Code 待整理/`
- `ai_fashion_mutually_exclusive_blip_v7_ipynb_的副本.py` — V7 主流程(模型/訓練/評估/索引/demo)
- `mater_kai_datascienctist.py` — 資料準備 Route A(下載/打 TAR/推 HF/FAISS)
- `處理資料集part2.py` — 資料準備 2 + 標記 UI + 多標籤分類器
- `驗證一下master_kai_的成果.py` — 抽向量 + 載 artifacts + Gradio demo

> 重構原則:把這些 notebook 的**邏輯**搬進 `src/aifashion/` 的乾淨模組(去重複、設定分離、
> 機密外置);Colab/Drive 專用的執行碼留在 notebook 端。

# AI Fashion Recommender 👗🔎

> 用一句**自然語言**描述你想要的穿搭,系統用 **CLIP 跨模態檢索**從服飾庫推薦最相符的 Top-K,並標註 cosine 相似度。

以 OpenAI **CLIP(ViT-B-32)** 的影像/文字共用嵌入空間做「文字 → 服飾」檢索;支援 **Zero-shot** 與 **Finetuned** 兩種模式,並用 **R@K** 量化效能。資料層是一條**雙雲(Google Drive + Hugging Face Hub)版本化 artifact pipeline**。

---

## ✨ 專案亮點(這兩個是面試主打)

1. **模型迭代:從 R@10 0.009 → 0.706 的除錯故事**
   診斷出 CLIP 微調時的 **embedding 空間漂移(space drift)**,改用「**把文字端固定為 anchor、只訓練影像投影頭去對齊**」+ **BLIP 強化 caption**,把檢索指標從幾乎無效拉到可用區間。

2. **資料工程:雙雲版本化 artifact pipeline**
   **Google Drive(快取/工作區)+ Hugging Face Hub(不可變、版本化的 Source-of-Truth)**;去重/清理/正規化 → WebDataset 分片 → 產 artifacts → 版本化發佈 → 自動驗證 → 更新 `latest` 別名 → 下游只讀 `latest`。這是 feature store / model registry 的 pattern。

---

## 🧩 問題與方法

時尚搭配難用關鍵字描述,且服飾庫龐大。做法:用 CLIP 把圖片與文字投影到**同一個向量空間**,查詢時把使用者文字編碼成向量,在影像向量索引中找**最近鄰**即為推薦。

- **Zero-shot**:直接用原生 CLIP 的影像/文字向量(I0–T0)。
- **Finetuned**:在 CLIP 之上訓練一個**影像投影頭**,使影像向量更貼近文字 anchor(I1–T0)。

---

## 📈 實驗結果(Fashionpedia,R@10)

| 版本 | 做法 | R@10 |
|---|---|---|
| V5 Baseline | 直接 CLIP zero-shot;caption 太弱、座標漂移 | ~0.009 |
| V6 互斥詞 | 對比學習 + label smoothing + 互斥詞去噪 | 0.03–0.12 |
| **V7 (BLIP + Anchor-to-T0)** | **文字固定為 anchor、只訓影像頭;BLIP + 屬性短詞強化 caption** | **Zero-shot 0.706 / Finetuned 0.674** |

**結論**:先把文字固定成語義 anchor,再讓影像去對齊;搭配強 caption,指標自然回到可用區間。

---

## 🏗️ 系統架構(兩條 pipeline)

### A) 資料 / 雲端 pipeline（花最多心力的部分）
```
Fashionpedia (Kaggle/HF)
   └─ 去重 / 清理 / 正規化
        └─ WebDataset 分片 (.tar + manifest.json)        ← 大資料用「分片 + 串流」,不全載進記憶體
             └─ 產出 artifacts: vecs.npz / index.faiss / keys.npy / meta.json / stats.json
                  └─ 版本化發佈到 HF Hub (Source of Truth)   ← 不可變、可回溯
                       └─ 自動驗證 (run checks)
                            └─ 更新 `latest` 別名            ← 解耦「產出者」與「消費者」
                                 └─ 消費者只讀 `latest`: ① Gradio demo ② 正式評估 ③ API
```
- **雙雲分工**:Drive 當可變快取/工作區(下載、暫存、續傳);HF Hub 當不可變、版本化的 SoT。
- **韌性**:批次下載**可續傳、冪等**(只補缺少的前綴,直到缺口為 0)。

### B) 模型 pipeline
```
WDS 串流 → CLIP(ViT-B-32) backbone → 影像投影頭(訓練) → R@K 評估 → 建索引 → Gradio demo(KG / BLIP-ITM 重排序)
```

> 架構圖見 `docs/architecture.png`(從專題簡報匯出)。

---

## 📁 專案結構
```
ai-fashion-recommender/
├── README.md
├── requirements.txt / .gitignore
├── configs.py                  # 集中設定(模型/訓練/路徑/HF;機密走環境變數)
├── src/aifashion/
│   ├── data/
│   │   ├── prepare.py          # 下載 + 去重/清理/正規化 + 打 WebDataset 分片
│   │   ├── captions.py         # heuristic + BLIP captioning、cap_map
│   │   ├── webdataset.py       # 訓練用的 robust WDS 串流
│   │   └── hub.py              # 雙雲 artifacts + HF Hub 版本化發佈 + `latest` 別名 + 驗證
│   ├── model.py                # CLIP backbone + 影像投影頭(anchor-to-T0)
│   ├── train.py                # 訓練影像頭對齊文字 anchor
│   ├── eval.py                 # R@K(zero-shot vs finetuned)
│   ├── index.py                # 建近鄰索引(FAISS / HNSW)
│   └── rerank.py               # 屬性 / 知識圖譜 / BLIP-ITM 重排序
├── app/demo.py                 # Gradio 互動式 demo
├── notebooks/                  # 原始 Colab notebook(需 GPU/Drive,僅供重現)
└── docs/architecture.png
```

---

## 🚀 怎麼跑
```bash
pip install -r requirements.txt
export HF_TOKEN=...            # 讀私有 HF repo 才需要;機密只走環境變數
# 訓練 / 評估 / demo(模組搬完後)
python -m aifashion.train
python -m aifashion.eval
python app/demo.py
```
> 註:資料/雲端 pipeline 需 Colab + GPU + Drive 環境執行(見 `notebooks/`);本機主要跑評估與 demo。

---

## 🛠️ 技術棧
PyTorch · open_clip(CLIP ViT-B-32) · BLIP(transformers) · WebDataset · FAISS / hnswlib · Gradio · Hugging Face Hub · Google Drive

## ⚠️ 限制與未來工作
- 資料/雲端 pipeline 目前綁 Colab/Drive;未來可抽成雲端無關的 CLI。
- 索引為近似最近鄰(ANN),大規模時需評估 recall/延遲取捨。
- 可加入更強 backbone(ViT-L/14-336)與線上 A/B 評估。

## 👥 致謝
第二屆半導體 AI 與 ChatGPT 應用班・第二組「AI Fashion 推薦穿搭系統」。組長:范希凱;指導:張志勇老師、蒯思齊老師。

> 本 repo 由原始 Colab notebook 重構為 production-style 套件(模組化、去重複、設定分離、機密外置)。

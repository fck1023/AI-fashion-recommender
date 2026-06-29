# AI Fashion Recommender 👗🔎

> 用一句**自然語言**描述你想要的穿搭,系統用 **CLIP 跨模態檢索**從服飾庫推薦最相符的 Top-K,並標註 cosine 相似度。

以 OpenAI **CLIP(ViT-B-32)** 的影像/文字共用嵌入空間做「文字 → 服飾」檢索;支援 **Zero-shot** 與 **Finetuned** 兩種模式,並用 **R@K** 量化效能。資料層是一條 **雙雲(Google Drive + Hugging Face Hub)版本化 artifact pipeline**。

---

## ✨ 專案亮點(這兩個是面試主打)

1. **模型除錯 + 誠實 ablation:R@10 0.009 → 0.706**
   診斷出 V5/V6 微調時的 **embedding 空間漂移(space drift)**——影像端與文字端一起動、互相遷就,R@10 崩到 **0.009**。修法:**anchor-to-T0**(凍結 CLIP backbone、把文字端固定成 anchor、只訓練影像投影頭去對齊),再用 **BLIP 強化 caption**。
   **關鍵是我做了 zero-shot vs finetuned 的 ablation,誠實量出 zero-shot(0.706)其實 ≥ finetuned(0.674)——真正的 lift 來自 caption 品質,不是 fine-tuning。** 這個發現比「我訓出 SOTA」更能展示我會做嚴謹評估、講得清因果。

2. **資料工程:雙雲版本化 artifact pipeline**
   **Google Drive(可變快取/工作區)+ Hugging Face Hub(不可變、版本化的 Source-of-Truth)**;分批下載(可續傳、冪等)→ 扁平化/manifest → 打包 → **atomic commit 版本化發佈** → 自動驗證 → 更新 **`latest` 別名** → 下游(demo/eval/API)只讀 `latest`。這就是 feature store / model registry 的 pattern。

---

## 🧩 問題與方法

時尚搭配難用關鍵字描述,且服飾庫龐大。做法:用 CLIP 把圖片與文字投影到**同一個向量空間**,查詢時把使用者文字編碼成向量,在影像向量索引中找**最近鄰**即為推薦。

- **Zero-shot**:直接用原生 CLIP 的影像/文字向量(I0–T0)。
- **Finetuned**:在 CLIP 之上訓練一個**影像投影頭**,使影像向量更貼近固定的文字 anchor(I1–T0)。

---

## 📈 實驗結果(Fashionpedia,R@10)

| 版本 | 做法 | R@10 |
|---|---|---|
| V5 Baseline | CLIP zero-shot;caption 太弱、訓練時座標漂移 | ~0.009 |
| V6 互斥詞 | 對比學習 + label smoothing + 互斥詞去噪 | 0.03–0.12 |
| **V7(BLIP + Anchor-to-T0)** | **文字固定為 anchor、只訓影像頭;BLIP 強化 caption** | **Zero-shot 0.706 / Finetuned 0.674** |

### 🔬 誠實發現(ablation 才是重點)

- **anchor-to-T0 達成的是「止血」**:它阻止了 V5/V6 的空間漂移(R@10 不再崩到 0.009),finetuned 回到健康的 0.674。
- **但 fine-tuning 沒有贏過 zero-shot**(0.674 < 0.706)。原因:目標 anchor 來自 CLIP 自己已對齊的文字空間、加上只有一層線性影像頭,**幾乎沒有 headroom 可學**。
- **真正把指標從 0.009 拉到 0.7 的,是「資料 / caption 品質」**:BLIP 生成式 caption 取代了「資料夾名+檔名」的弱啟發式(見 `data/captions.py` 的 `resolve_caption` 覆寫鏈)。
- 結論一句話:*V5 的 0.009 不是 CLIP 不行,是 caption 太爛、又把嵌入空間訓壞了;把這兩件事修好,zero-shot 就有 0.706。*

---

## 🏗️ 系統架構(兩條 pipeline)

### A) 資料 / 雲端 pipeline(花最多心力的部分)
```
Fashionpedia(HF dataset)
  └─ prepare:分批下載(可續傳/冪等)→ 扁平化 XX/ 分桶 → items.csv·paths.json(manifest)→ 打單一 .tar
       └─ hub.verify_dataset:抽樣開圖 / 壞檔 / 重複 → verify_report.json
            └─ hub.publish_dataset:TAR + manifest + 報告 → atomic commit 上 HF Hub(SoT)
  ── 模型產出側 ──
  index:抽影像向量 → vecs.npz · index_ip.faiss · index_keys.npy · index_meta.json
       └─ hub.publish_artifacts:date-stamp 版本化 → 更新 `latest` 別名
            └─ 消費者只讀 `latest`:① Gradio demo  ② R@K 評估  ③ API
```
- **雙雲分工**:Drive 當可變快取/工作區(下載、暫存、續傳);HF Hub 當不可變、版本化的 SoT。
- **韌性**:批次下載 `resume_download` 可續傳;`rsync --ignore-existing` 冪等;429 自動降速重試。

### B) 模型 pipeline
```
WDS 串流(robust)→ CLIP(ViT-B-32) backbone(凍結)→ 影像投影頭(訓練,對齊文字 anchor)
   → R@K 評估(zero-shot vs finetuned)→ 建索引(FAISS/HNSW)→ Gradio demo
                                                              (兩階段:向量粗排 → BLIP-ITM 精排)
```

> 架構圖見 `docs/architecture.png`(從專題簡報匯出)。

---

## 📁 專案結構
```
ai-fashion-recommender/
├── README.md  ·  requirements.txt  ·  pyproject.toml  ·  .gitignore
├── src/aifashion/
│   ├── configs.py             # 集中設定(模型/訓練/路徑/HF;機密走環境變數)
│   ├── model.py               # AnchoredCLIP:backbone + 影像投影頭(anchor-to-T0)
│   ├── train.py               # info_nce_anchor_loss + 訓影像頭(AMP 迴圈)
│   ├── eval.py                # recall_at_k + zero-shot vs finetuned 報告
│   ├── index.py               # 抽向量 → FAISS/HNSW → 存/載 artifacts
│   ├── rerank.py              # 兩階段 retrieve→rerank(屬性零樣本 + BLIP-ITM;KG 為擴充點)
│   └── data/
│       ├── prepare.py         # 分批下載 + 扁平化(hex 分桶)+ manifest + 打 TAR
│       ├── captions.py        # caption 解析鏈 + BLIP 生成 + cap_map 覆寫
│       ├── webdataset.py      # robust WDS 串流(Drive 優先、HF 保底)
│       └── hub.py             # 雙雲版本化發佈(atomic commit)+ 驗證 + `latest` 別名
├── app/demo.py                # Gradio 兩分頁互動 demo(Text→Image / Image→Image)
├── scripts/                   # CLI 進入點(把上面函式串成一鍵指令)
├── notebooks/                 # 原始 Colab notebook(需 GPU/Drive,僅供重現對照)
└── docs/architecture.png
```

---

## 🚀 怎麼跑
```bash
pip install -e .              # src-layout 套件,裝完即可 import aifashion
export HF_TOKEN=...           # 讀私有 HF repo 才需要;機密只走環境變數

python app/demo.py            # 啟動 Gradio demo(需先有 index artifacts:本地 runs/ 或 HF `latest`)
```

Python API(訓練 / 評估 / 建索引 / 發佈 —— 需 GPU + 資料):
```python
from aifashion.model import AnchoredCLIP
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.train import train_image_head
from aifashion.eval import evaluate
from aifashion.index import encode_images_to_vectors, save_artifacts
from aifashion.data.hub import publish_artifacts

m = AnchoredCLIP.from_pretrained(device="cuda")
train, val, source = build_streams(prefer="drive")              # Drive 找不到 shards 自動回退 HF
train_loader = make_loader(train, m.preprocess_train, m.tokenizer)
val_loader   = make_loader(val,   m.preprocess_eval,  m.tokenizer)

train_image_head(m, train_loader, val_loader)                   # 印訓前/後 R@K,存影像頭
evaluate(m, val_loader)                                         # zero-shot vs finetuned → compare.json

vecs, keys = encode_images_to_vectors(m, val_loader)            # 預設 zero-shot 向量(ablation 顯示最佳)
run_dir = save_artifacts(vecs, keys)                            # 產 4 個 artifact 檔
publish_artifacts(run_dir, "Kai1023/kai-outfit-ai-dataset")    # 版本化發佈 + 更新 latest
```
> 完整資料/雲端 pipeline(Route A 下載/打包/發佈)需 Colab + GPU + Drive,見 `notebooks/`;本機主要跑 demo 與評估。

---

## 🛠️ 技術棧
PyTorch · open_clip(CLIP ViT-B-32) · BLIP(transformers) · WebDataset · FAISS / hnswlib · Gradio · Hugging Face Hub · Google Drive

## ⚠️ 限制與未來工作
- fine-tuning 目前沒贏過 zero-shot;要讓它有意義,需換更難的對齊目標(非 CLIP 自身文字空間)、加大投影頭容量、或導入 hard negatives。
- 資料/雲端 pipeline 仍偏 Colab/Drive;`scripts/` 正逐步抽成雲端無關的一鍵 CLI。
- 索引為近似最近鄰(ANN),大規模時需評估 recall/延遲取捨;可加更強 backbone(ViT-L/14-336)與線上 A/B。

## 👥 致謝
第二屆半導體 AI 與 ChatGPT 應用班・第二組「AI Fashion 推薦穿搭系統」。組長:范希凱;指導:張志勇老師、蒯思齊老師。

> 本 repo 由原始 Colab notebook(約 8,780 行)重構為 production-style 套件:模組化、去重複、設定與機密分離、忠實保留實際執行邏輯並標註改進處。

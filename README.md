# AI Fashion Recommender 👗🔎

> 用一句**自然語言**(例如「black leather jacket」)描述你想要的穿搭,系統用 **CLIP 跨模態檢索**從 4 萬張服飾庫裡找出最相符的 Top-K,並標註相似度。

---

## 一分鐘看懂這個專案

時尚很難用關鍵字精準描述,服飾庫又很大。做法:用 OpenAI **CLIP(ViT-B-32)** 把「圖片」和「文字」投影到**同一個向量空間**;查詢時把使用者的文字編成向量,在影像向量索引中找**最近鄰**,就是推薦結果。

整個專案分兩條主線:
- **模型線**:CLIP 檢索 + 一個「只訓影像投影頭」的微調設計(anchor-to-T0)+ 用 R@K 嚴謹評估。
- **資料線**:一條 **雙雲(Google Drive + Hugging Face Hub)版本化資料管線**——這是整個專案投入最多、也最像實際工程的部分。

---

## ✨ 三個最值得看的亮點

1. **一場誠實的 ablation:caption 品質才是關鍵,不是模型微調。**
   我用三組對照量化證明了「lift 從哪來」(數字見下表)。這比「我訓出了 SOTA」更能展現我**會做嚴謹評估、講得清因果**。

2. **雙雲版本化資料管線(feature-store / model-registry 的 pattern)。**
   Drive 當「可變的快取/工作區」、HF Hub 當「不可變、可回溯的 Source-of-Truth」;每次發佈是一個 atomic commit,再用 `latest` 別名解耦產出者與消費者。

3. **把 8,780 行 ChatGPT 散裝 notebook 重構成 production-style 套件。**
   模組化、去重複、設定與機密分離,並在過程中**找出並修掉原碼的真實 bug**(AMP 下漏 `scaler.unscale_`、評估缺 `no_grad`、KG 計分自相矛盾等)。

---

## 📈 成果:三段式 ablation(真實重現,R@K @ 1500 樣本)

| 設定 | 用什麼 caption / 模型 | R@1 | R@5 | **R@10** |
|---|---|---|---|---|
| 啟發式 caption | 檔名雜湊 `"fa8c59 outfit"`(無語義) | 0.001 | 0.002 | **0.007** |
| **BLIP caption,zero-shot** | BLIP 真描述 + 原生 CLIP(**零訓練**) | 0.332 | 0.575 | **0.675** |
| BLIP caption,fine-tuned | 同上 + 我訓練的影像投影頭 | 0.198 | 0.410 | **0.519** |

**這張表就是整個專案的結論:**

- 🟢 **caption 就是一切**:同一批圖、同一個 CLIP、**完全不訓練**,只把「檔名雜湊」換成「BLIP 真描述」,R@10 從 **0.007 → 0.675**(×96 倍)。
- 🔴 **我的 fine-tuning 反而扣分**:zero-shot **0.675 > finetuned 0.519**。微調那層影像頭把檢索拉低了 ~0.15。
- 🧭 **所以結論是**:這個專案的價值來自**資料 / caption 工程**,不是模型微調——這也是為什麼我把最多力氣放在那條雙雲 caption 管線上。

> 歷史脈絡:V5 baseline R@10 ≈ 0.009(caption 太弱 + 微調把嵌入空間訓壞)、V6 加互斥詞去噪 ≈ 0.03–0.12、V7(本版,BLIP + anchor-to-T0)zero-shot 回到 ~0.7。上表是 V7 在 1500 樣本上的重現,故事與原報告一致。

> 📐 **量測嚴謹說明**:R@K 在 **gallery 大小 N=1500** 下量測;此指標**隨 N 單調下降**(候選池越大越難命中),所以**只有相同 N 才能互相比較**——上表三列都在同一 N、同一批樣本下,差異才完全來自 caption / 模型。評估固定落在「有 BLIP caption 覆蓋的子集」(`cap_map` ~3008 筆)內,確保每個 query 都有有效文字;**產品端的檢索 / demo 則使用全部 45,623 張影像的索引**(gallery 影像不需 caption 即可被檢索)。

---

## 🧠 核心設計:anchor-to-T0(怎麼修掉 embedding 漂移)

V5/V6 同時微調影像端與文字端,兩邊座標一起動、互相遷就 → **embedding 空間漂移**,檢索學不到穩定語義(R@10 崩到 0.009)。V7 的修法:

1. **凍結 CLIP backbone**(只當特徵抽取器)。
2. **把文字端固定成 anchor**:`txt_proj` 凍結為 Identity,文字向量恆等於 CLIP 原生 T0,當作不動的語義座標。
3. **只訓練影像投影頭** `img_proj`,讓影像向量去對齊 T0。
4. 兩個頭都以 **Identity 初始化** → 從「I1 ≈ I0」這個乾淨起點出發。

> 它成功「止血」(不再漂移),但 ablation 顯示:因為目標 anchor 來自 CLIP 自己已對齊的文字空間、加上只有一層線性頭,**沒有 headroom 可贏過 zero-shot**。誠實面對這點,正是這個專案的價值。

---

## 🔄 端到端工作流(資料怎麼一路變成推薦)

```
[1] 備資料        [2] 生caption        [3] 串流        [4] (可選)訓練     [5] 評估        [6] 建索引+發佈        [7] Demo
 prepare ──► HF Hub   captions ──► cap_map   webdataset ──► (model+train) ──►  eval ──►  index ──► HF latest ──► app/demo + rerank
```

1. **備資料(`data/prepare.py` + `data/hub.py`)** — 從 HF 分批下載 4 萬張圖到 Drive(可續傳)→ 扁平化成 `XX/檔名` → 產 manifest(items.csv / paths.json)+ 打 TAR → 驗證(抽樣開圖 / 壞檔 / 重複)→ atomic commit 版本化上 HF Hub。
   *為什麼:* 原始資料雜亂、量大、下載會斷;這步讓資料「可重現、可回溯」。
2. **生 caption(`data/captions.py`)** — 用 **BLIP** 對每張圖生成描述,寫成 `cap_map.jsonl`。
   *為什麼:* 檔名是雜湊(`fa8c59.jpg`)毫無語義;**caption 品質直接決定檢索成敗**(就是上表 0.007 → 0.675 的關鍵)。
3. **串流讀取(`data/webdataset.py`)** — 訓練/評估時從 shards 串流讀圖,套用 caption 解析鏈:`cap_map` 覆寫弱字、丟掉噪聲 caption。
   *為什麼:* 大資料不全載進記憶體;單一壞檔不讓整個 epoch 掛掉。
4. **(可選)訓練(`model.py` + `train.py`)** — 凍結 CLIP,只訓一層影像投影頭去對齊文字 anchor(anchor-to-T0)。
   *為什麼:* 避免整個微調造成的漂移;但 ablation 顯示這步 lift 有限。
5. **評估(`eval.py`)** — zero-shot vs finetuned 的 R@1/5/10 → `compare.json`。
   *為什麼:* R@K 是檢索主指標;這支讓「caption / 訓練到底有沒有用」可量化、可重現。
6. **建索引 + 發佈(`index.py` + `data/hub.py`)** — 抽全庫影像向量 → 建 FAISS 索引 → date-stamp 版本化上傳 → 更新 `latest` 別名。
   *為什麼:* 推薦 = 在向量索引找最近鄰;`latest` 讓下游換版零改動。
7. **Demo / 檢索(`app/demo.py` + `rerank.py`)** — 載 `latest` 索引 → 文字/圖片查詢 → 向量粗排 →(可選)BLIP-ITM 交叉編碼精排 → Top-K 縮圖。
   *為什麼:* 雙塔向量檢索快但精度有上限,交叉編碼準但貴 → 只對少量候選做精排。

---

## 📂 每支檔案做什麼、為什麼存在

| 檔案 | 做什麼 | 為什麼存在(解決什麼問題) |
|---|---|---|
| `src/aifashion/configs.py` | 所有可調參數(模型 / 訓練 / 路徑 / HF repo)用 dataclass 集中一處 | 原 notebook 參數散在各 cell、改一個要翻好幾頁還前後不一;機密(HF_TOKEN)只走環境變數、不進 git |
| `src/aifashion/model.py` | `AnchoredCLIP`:凍結 backbone + 可訓練影像頭 + 凍結文字 anchor 頭 | 把專案核心設計(anchor-to-T0)封裝成乾淨 `nn.Module`;與原始 checkpoint 格式相容 |
| `src/aifashion/train.py` | 對比損失 `info_nce_anchor_loss` + 只訓影像頭的 AMP 迴圈,印訓前/後 R@K | 收斂散落的實驗訓練碼;**修掉原碼在 AMP 下漏 `scaler.unscale_` 的 bug** |
| `src/aifashion/eval.py` | `recall_at_k` + zero-shot vs finetuned 報告 → `compare.json` | R@K 是檢索主指標;讓 caption/訓練的效果可量化、可重現(**修掉缺 `no_grad` 的 bug**) |
| `src/aifashion/index.py` | 抽全庫影像向量 → FAISS / HNSW 索引 → 存/載 4 個 artifact | 推薦 = 向量最近鄰;產出的檔正是 `hub` 上傳的內容(索引層↔發佈層閉環) |
| `src/aifashion/rerank.py` | 兩階段 retrieve→rerank(向量粗排 → BLIP-ITM 精排)+ 零樣本屬性打分 | 取代舊 Cell 9 那坨「加權雜湊湯」,改成每段分數都有明確意義的工業界標準形狀 |
| `src/aifashion/data/prepare.py` | 分批下載(可續傳)→ 扁平化 hex 分桶 → manifest → 打 TAR | 把 4 萬張雜亂原圖整理成可重現的資料集;**收斂原碼重複 5 次的 `norm_rel`** |
| `src/aifashion/data/captions.py` | caption 解析鏈(json→txt→啟發式→`cap_map` 覆寫→噪聲過濾)+ BLIP 生成 | **檢索 lift 的真正來源**;把「弱檔名字」換成「BLIP 真描述」 |
| `src/aifashion/data/webdataset.py` | robust WDS 串流(Drive 優先、HF 保底),產 `(images, tokens, captions, keys)` | 大資料串流不爆記憶體;**改進:影像解碼包 try/except,壞檔不讓 epoch 掛掉** |
| `src/aifashion/data/hub.py` | 雙雲版本化發佈(atomic commit)+ 資料驗證 + `latest` 別名 | HF Hub 當不可變 SoT;`latest` 解耦產出者/消費者(feature-store / model-registry pattern) |
| `app/demo.py` | Gradio 兩分頁(Text→Image / Image→Image)互動檢索 demo | 整條 pipeline 的「消費端」:載 artifacts → 查詢 → Top-K 縮圖拼貼 |
| `scripts/smoke_test.py` | 不需資料/GPU 的 dry-run,4–5 階段驗證整包跑得起來 | 接真資料訓練前,先一行確認「整包沒壞」 |
| `scripts/run_*.py` | `run_prepare / run_train / run_eval / run_index` 一鍵 CLI | 把套件函式串成可直接執行的指令,接真資料就是一行 |
| `notebooks/colab_run.py` | 自含單檔 Colab 腳本(WDS + caption + 訓練 + 評估) | 不需安裝套件、貼一個 cell 就能在 Colab 跑真資料、看真 R@K |
| `configs / pyproject.toml` | `pip install -e .` 後即可 `import aifashion`(src-layout) | 讓套件可安裝、可被 `app/`、`scripts/` 乾淨匯入 |

---

## 🚀 怎麼跑

**A. 最快看到真實結果(Colab,推薦)** — 用自含單檔,不必安裝套件:
```text
1. Colab 執行階段 → 變更類型 → GPU(T4)
2. 把 notebooks/colab_run.py 整支貼進一個 cell
3. 改最上面 CONFIG 的 SRC_DIR(你的 .tar shards)、RUN_DIR(有 cap_map.jsonl 的資料夾)
4. 執行 → 看 zero-shot vs finetuned 的 R@K
```

**B. 當套件用(本機 / 伺服器)**:
```bash
pip install -e .                 # src-layout,裝完即可 import aifashion
export HF_TOKEN=...              # 讀私有 HF repo 才需要;機密只走環境變數
python scripts/smoke_test.py --model --train   # dry-run:確認整包能跑(免資料/GPU)
python app/demo.py               # 啟動 Gradio demo(需先有 index artifacts)
```

**C. Python API**:
```python
from aifashion.model import AnchoredCLIP
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.eval import evaluate

m = AnchoredCLIP.from_pretrained(device="cuda")
_, val, source = build_streams(prefer="drive", shards_dir="/path/to/shards")
val_loader = make_loader(val, m.preprocess_eval, m.tokenizer)
evaluate(m, val_loader)          # zero-shot vs finetuned R@K → compare.json
```

---

## 🛠️ 技術棧
PyTorch · open_clip(CLIP ViT-B-32)· BLIP(transformers)· WebDataset · FAISS / hnswlib · Gradio · Hugging Face Hub · Google Drive

## ⚠️ 限制與未來工作
- **fine-tuning 目前沒贏過 zero-shot**;要讓它有意義,需換更難的對齊目標(非 CLIP 自身文字空間)、加大投影頭容量、或導入 hard negatives。
- 資料/雲端管線仍偏 Colab/Drive;`scripts/` 正逐步抽成雲端無關的一鍵 CLI。
- 索引為近似最近鄰(ANN),大規模時需評估 recall/延遲取捨;可加更強 backbone(ViT-L/14-336)與線上 A/B 評估。

## 👥 致謝
第二屆半導體 AI 與 ChatGPT 應用班・第二組「AI Fashion 推薦穿搭系統」。組長:范希凱;指導:張志勇老師、蒯思齊老師。

> 本 repo 由原始 Colab notebook(約 8,780 行)重構為 production-style 套件:模組化、去重複、設定與機密分離、忠實保留實際執行邏輯並標註改進處。原始 notebook 留在 `notebooks/` 供「重構前 vs 後」對照。

# AI Fashion Recommender 👗

用一句話描述你想要的穿搭(例如 *black leather jacket*),從四萬多張服飾照片裡找出最相符的幾張。

底層用的是 OpenAI 的 **CLIP**——它把圖片和文字放進同一個向量空間,所以「用文字找圖」就變成「在向量裡找最近鄰」這件單純的事。這個 repo 是我把一份課程專題,從「ChatGPT 協助寫成、跑得動但結構雜亂的 Colab(約 8,780 行)」,整理成一個結構清楚、自己完全理解、也能拿出來給人看的專案。

---

## 我最想講的一件事:真正有用的不是模型,是資料

一開始我以為重點是「把模型訓練好」。後來我做了一組對照實驗,結論幾乎是相反的。

同一批圖、同一個 CLIP,我只改了一件事——把每張圖的文字描述,從「檔名」換成「BLIP 自動生成的句子」:

| 給模型的文字描述 | 有沒有額外訓練 | R@10 |
|---|---|---|
| 檔名雜湊(`fa8c59 outfit`,沒有語義) | 否 | **0.007**(≈ 隨機水準) |
| BLIP 生成的真描述(`a woman in a white dress`) | 否(原生 CLIP) | **0.675** |
| BLIP 生成的真描述 | 有(我訓練的投影頭) | **0.519** |

光是把文字描述從「無語義的檔名」換成「BLIP 生成的真實描述」,R@10 就從 0.007 跳到 0.675。而我額外訓練的那層投影頭,不但沒幫上忙,反而把分數拉低到 0.519。

所以這個專案真正的價值,不在模型微調,而在**資料與 caption 的品質**。能用數字證明這件事、並誠實承認「我的微調其實是反效果」,比硬說「我訓出了很強的模型」有說服力得多——這也是為什麼我後來把大部分力氣,都投在那條資料管線上。

> 數字怎麼讀:R@K 是在 **gallery 大小 N=1500** 下量的,這個指標會隨 N 變大而下降(候選越多越難命中),所以只有相同 N 才能互相比較;上表三列都在同一 N、同一批樣本下,差異才完全來自「描述」與「模型」。評估固定落在有 BLIP 描述覆蓋的子集內,確保每個查詢都有有效的文字;至於 demo 的實際檢索,用的是全部 45,623 張圖的索引(被檢索的圖不需要描述也能被找出來)。

---

## 我怎麼修掉「越訓練越糟」的問題

更早的版本(V5/V6)我是「影像端和文字端一起微調」,結果兩邊座標一起跑、互相遷就,整個嵌入空間漂掉,R@10 一度崩到 0.009。

這版(V7)的修法,我把它叫做 **anchor-to-T0**:

1. 把 CLIP 整個凍結,只當特徵抽取器。
2. 把**文字端固定成不動的錨點**——文字向量就用 CLIP 原生的,當作不會跑的語義座標。
3. 只訓練一層**影像投影頭**,讓影像向量去對齊那個固定的文字錨點。

它確實「止血」了(空間不再漂),但如同上面的數字所示,它沒能真的贏過完全不訓練的版本。原因也想得通:文字錨點本來就出自 CLIP 已經對齊好的空間,加上我只給了一層線性投影,根本沒有多少「進步空間」可學。

---

## 我花最多時間的地方:雙雲資料管線

四萬張圖怎麼下載、清理、版本化保存、再餵進訓練——這塊才是我學到最多 system design 的地方。設計上我讓兩朵雲各司其職:

- **Google Drive** 當「可變的工作區」:分批下載、暫存、續傳。下載會斷、會被限流,所以這層要能**從斷掉的地方接著跑、重跑也不會重抓**。
- **Hugging Face Hub** 當「不可變的真相來源(source of truth)」:每次發佈是一個 atomic commit,要回溯哪一版資料、哪一版索引都查得到。

發佈流程是:整理好資料 → 打包 → 自動健檢(抽樣開圖、找壞檔、數重複)→ 版本化上傳 → 更新一個 `latest` 指標。下游(demo、評估、API)永遠只讀 `latest`,所以換新版只要動那個指標、下游零改動——這其實就是 feature store / model registry 在做的事,我在這個專案裡把它走了一遍。

---

## 程式是怎麼組織的

我把原本擠在一起的 notebook,拆成三層、各司其職:

```
src/aifashion/
├── configs.py        所有可調參數集中一處(模型、訓練、路徑、HF repo),機密只走環境變數
│
├── 資料層 data/
│   ├── prepare.py    下載 → 扁平化分桶 → 產 manifest → 打包;把雜亂原圖變成可重現的資料集
│   ├── captions.py   caption 的解析鏈 + BLIP 生成;這支是上面那個 0.007→0.675 的關鍵
│   ├── webdataset.py 串流讀圖(不全載進記憶體),單一壞檔不會讓整個 epoch 掛掉
│   └── hub.py        雙雲版本化發佈 + 資料健檢 + latest 指標
│
├── 模型層
│   ├── model.py      AnchoredCLIP:凍結的 backbone + 可訓練影像頭 + 凍結的文字錨點
│   ├── train.py      對比損失 + 只訓影像頭的訓練迴圈(會印訓練前後的 R@K 對照)
│   ├── eval.py       R@K 計算 + zero-shot vs finetuned 報告
│   ├── index.py      抽全庫向量 → 建 FAISS/HNSW 索引 → 存/載 artifacts
│   └── rerank.py     兩階段檢索:向量粗排 → BLIP-ITM 交叉編碼精排
│
└── 服務層
    ├── app/demo.py            Gradio 互動 demo(文字找圖 / 圖找圖)
    ├── scripts/               一鍵 CLI:smoke_test 先確認整包能跑,再 prepare/train/eval/index
    └── notebooks/colab_run.py 自含單檔,貼進 Colab 一個 cell 就能用真資料跑出真 R@K
```

整理的過程裡,我也讀懂並修掉了原本 ChatGPT 版本的幾個真實 bug:訓練在混合精度下漏了梯度 unscale、評估忘了關梯度追蹤、還有舊版 UI 把好幾種不同量綱的分數混加成一個可能超過 1、又自相矛盾的分數。能指出「哪些是原作、哪些是我改的、為什麼要改」,是我覺得這次重構最有價值的部分。

---

## 怎麼跑

**想最快看到真實結果(Colab):** 把 `notebooks/colab_run.py` 整支貼進一個 cell,改最上面的 `SRC_DIR`(你的 .tar 資料夾)和 `RUN_DIR`(放 caption 的資料夾),執行就好——不必安裝套件。

**當成套件用:**

```bash
pip install -e .
python scripts/smoke_test.py --model --train   # 不用資料/GPU,先確認整包跑得動
python app/demo.py                              # 起 Gradio demo(需先有索引 artifacts)
```

**用 Python API:**

```python
from aifashion.model import AnchoredCLIP
from aifashion.data.webdataset import build_streams, make_loader
from aifashion.eval import evaluate

m = AnchoredCLIP.from_pretrained(device="cuda")
_, val, _ = build_streams(prefer="drive", shards_dir="/path/to/shards")
val_loader = make_loader(val, m.preprocess_eval, m.tokenizer)
evaluate(m, val_loader)        # 印 zero-shot vs finetuned 的 R@K
```

---

## 如果重來 / 還沒做的

- 微調目前贏不過 zero-shot;要讓它有意義,得換更難的對齊目標、加大投影頭容量,或導入 hard negatives。
- 評估的 caption 只覆蓋了一部分資料;要在更大樣本上量乾淨的 R@K,得先把更多圖補上 BLIP 描述。
- 資料管線目前還偏 Colab/Drive,正在慢慢抽成跟雲端無關的一鍵 CLI。

## 技術棧

PyTorch · open_clip(CLIP ViT-B-32)· BLIP · WebDataset · FAISS / hnswlib · Gradio · Hugging Face Hub · Google Drive

## 一點背景

這原本是「第二屆半導體 AI 與 ChatGPT 應用班・第二組」的專題。把當時跑得出結果、但很難維護的 notebook,重構成現在這個模組化、設定與機密分離、保留實際邏輯並標註改動的版本;原始 notebook 留在 `notebooks/` 供「重構前後」對照。

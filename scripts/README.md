# scripts/

一鍵 CLI 進入點:把 `src/aifashion/` 的函式串成可直接執行的指令。先 `pip install -e .`。

| 指令 | 做什麼 | 對應模組 |
|---|---|---|
| `python scripts/smoke_test.py [--model] [--train]` | **Dry-run**:不需資料/GPU,驗證整包跑得起來 | 全部 |
| `python scripts/run_prepare.py --repo-id <id> [--publish]` | 資料準備:下載→扁平化→manifest→TAR→驗證→發佈 | `data.prepare` · `data.hub` |
| `python scripts/run_train.py [--prefer hf] [--cap-map ...]` | 訓練影像投影頭(anchor-to-T0) | `train` · `data.webdataset` |
| `python scripts/run_eval.py [--heads ...]` | R@K 評估(zero-shot vs finetuned)→ compare.json | `eval` |
| `python scripts/run_index.py [--publish <id>]` | 抽向量→FAISS/HNSW→存/發佈 artifacts | `index` · `data.hub` |

典型流程(接真資料):
```bash
pip install -e .
python scripts/smoke_test.py --model --train           # 先確認整包沒壞
python scripts/run_prepare.py --repo-id <id> --publish # (Colab+Drive)備資料
python scripts/run_train.py --prefer drive --cap-map runs/v7/cap_map.jsonl
python scripts/run_eval.py  --heads runs/v7/last_extra.pt
python scripts/run_index.py --publish <id>             # 建索引並更新 HF latest
python app/demo.py                                     # 消費 latest 起 demo
```
共用慣例:超參不給就取 `configs.py` 預設;機密(HF_TOKEN)只走環境變數。

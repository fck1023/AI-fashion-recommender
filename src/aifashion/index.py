"""近鄰索引:把影像向量建成可快速檢索的索引 + 存/載 artifacts。

來源:mater_kai_datascienctist.py(抽向量 → npz → FAISS IndexFlatIP → keys/meta,
     行 1010–1080,取最完整的最後一版)+ 處理資料集part2.py 行 798(hnswlib 後端)。

──────────────────────────────────────────────────────────────────────────
設計重點(面試可講)
──────────────────────────────────────────────────────────────────────────
  - 推薦 = 在影像向量索引裡找「與 query 向量最近」的 Top-K。向量都 L2 normalize 過,
    所以內積(IP)== cosine 相似度,IndexFlatIP 直接當 cosine 檢索用。
  - 兩種後端:FAISS(快、精確)與 hnswlib(免 faiss、ANN 記憶體友善)擇一。
  - 產出的 4 個檔(vecs.npz / index_ip.faiss / index_keys.npy / index_meta.json)
    正是 hub.publish_artifacts 上傳的內容 → 索引層與雙雲發佈層在此閉環。
  - 抽向量預設用 zero-shot backbone(I0):本專案 ablation 顯示它檢索表現最好(見 README)。

備註(大規模的 System Design 考量,本檔留簡潔版):原 notebook 對 4 萬張用
np.memmap + progress.json 做「可續傳、不爆記憶體」的抽取;此處核心邏輯相同,
省略 checkpoint 細節以求可讀。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .configs import CFG

# 與 hub._ARTIFACT_NAMES 對齊的標準輸出檔名
NPZ_NAME = "vecs.npz"
FAISS_NAME = "index_ip.faiss"
KEYS_NAME = "index_keys.npy"
META_NAME = "index_meta.json"


def build_index(vectors: np.ndarray, backend: str = "faiss"):
    """以影像向量建立索引(faiss / hnswlib),回傳索引物件。向量須已 L2 normalize。"""
    vectors = np.asarray(vectors, dtype="float32")
    n, dim = vectors.shape
    if backend == "faiss":
        import faiss
        index = faiss.IndexFlatIP(dim)        # 向量已 L2 → 內積 = cosine
        index.add(vectors)
        return index
    if backend == "hnswlib":
        import hnswlib
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(vectors, np.arange(n))
        index.set_ef(max(50, 2 * 10))         # ef ≥ 查詢 K,給點餘裕
        return index
    raise ValueError(f"未知 backend:{backend}(支援 faiss / hnswlib)")


def search(index, query_vec: np.ndarray, k: int = 10):
    """查 Top-K,統一回傳 (indices, scores);scores 越大越相似(cosine)。"""
    q = np.asarray(query_vec, dtype="float32")
    if q.ndim == 1:
        q = q[None, :]
    if hasattr(index, "search"):              # faiss:回 (distances=IP, indices)
        scores, idx = index.search(q, k)
        return idx, scores
    labels, dists = index.knn_query(q, k=k)   # hnswlib cosine:距離 = 1 - cosine
    return labels, 1.0 - dists


def encode_images_to_vectors(model, loader, *, project: bool = False, max_n: int = 0):
    """掃 loader 抽全庫影像向量(L2 normalized)+ keys,回傳 (vecs[N,d] float32, keys[N])。

    project=False:用 zero-shot backbone 向量 I0(ablation 顯示檢索最好,預設)。
    project=True :用訓練後的投影向量 I1。
    """
    vecs, keys, n = [], [], 0
    for batch in loader:
        images, _, _, batch_keys = batch
        i0 = model.encode_image_features(images)
        v = model.project_image(i0) if project else i0
        vecs.append(v.cpu().numpy().astype("float32"))
        keys.extend(batch_keys)
        n += len(batch_keys)
        if max_n and n >= max_n:
            break
    return np.concatenate(vecs, 0), np.array(keys)


def save_artifacts(vectors: np.ndarray, keys, run_dir=None, *,
                   model_desc: str = "open_clip ViT-B-32 (openai)", extra: dict | None = None) -> Path:
    """把向量/索引/keys/meta 存成 4 個標準 artifact 檔(= hub.publish_artifacts 的輸入),回傳目錄。"""
    import faiss
    run_dir = Path(run_dir or CFG.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype="float32")
    keys = np.asarray(keys)

    np.savez(run_dir / NPZ_NAME, keys=keys, vecs=vectors)
    faiss.write_index(build_index(vectors, "faiss"), str(run_dir / FAISS_NAME))
    np.save(run_dir / KEYS_NAME, keys)

    meta = {
        "model": model_desc, "normalize": "L2", "metric": "IP",
        "n": int(vectors.shape[0]), "dim": int(vectors.shape[1]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    (run_dir / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Artifacts 已存:{run_dir}{[NPZ_NAME, FAISS_NAME, KEYS_NAME, META_NAME]}")
    return run_dir


def load_artifacts(run_dir=None):
    """載回索引消費端三件套:(faiss_index, keys, meta)。供 eval / demo / API 使用。"""
    import faiss
    run_dir = Path(run_dir or CFG.paths.run_dir)
    index = faiss.read_index(str(run_dir / FAISS_NAME))
    keys = np.load(run_dir / KEYS_NAME, allow_pickle=True)
    meta = json.loads((run_dir / META_NAME).read_text(encoding="utf-8"))
    return index, keys, meta

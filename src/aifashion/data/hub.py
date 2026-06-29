"""雙雲 artifact pipeline:Drive(可變快取/工作區)+ HF Hub(版本化 Source-of-Truth)。

來源:mater_kai_datascienctist.py
  - Route A(行 10–232):分批 snapshot_download → rsync 到 Drive(可續傳、冪等)
    → 打 TAR + items.csv/paths.json(manifest)→ 輕量驗證 → create_commit 推上 HF。
  - Artifacts(行 1082–1121):把 向量(.npz)/FAISS 索引/keys/meta 推到
    `artifacts/<model>/<date>/`(date-stamp 版本化)。

──────────────────────────────────────────────────────────────────────────
為什麼這支是「第二個面試主打」(feature store / model registry 的 pattern)
──────────────────────────────────────────────────────────────────────────
兩朵雲分工明確:
  - Google Drive = 可變的「快取 / 工作區」:下載暫存、續傳、跨 session 快啟動。
  - HF Hub       = 不可變、可回溯的「Source of Truth」:每次發佈 = 一個 atomic
                   commit,要回滾哪一版都查得到(= dataset / model versioning)。

發佈契約:產出檔 → 版本化上傳(atomic commit)→ 驗證 → 更新 `latest` 指標。
消費端(eval / demo / API)永遠只讀 `latest`,於是「產出者」與「消費者」解耦:
換新版只動 latest 指標、下游零改動。

誠實註記:date-stamp 版本化與 atomic commit 是**原始 notebook 就有**的;
`latest` 指標檔(update_latest/resolve_latest)是**本次重構新增**的改進
(原本只有日期資料夾、沒有顯式 latest 指標,下游得自己改路徑)。
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from ..configs import CFG

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
# 標準化後的 artifacts 檔名(index.py 會產這些;原 notebook 用較長的命名)
_ARTIFACT_NAMES = ("vecs.npz", "index_ip.faiss", "index_keys.npy", "index_meta.json")


def _hf_token() -> str | None:
    """HF token 只從環境變數讀(見 configs);公開 repo 可為 None。"""
    return CFG.hf_token or None


# ── 低階:把一組檔案做成一個 atomic 版本化 commit ───────────────────────────
def commit_files(repo_id: str, files: list[tuple[str, str | Path]], message: str, *,
                 repo_type: str = "dataset", revision: str = "main",
                 token: str | None = None) -> str:
    """files = [(repo 內路徑, 本機檔案), ...];一次 atomic commit,回傳 commit oid。

    原始 notebook 在好幾處重複貼了 `HfApi().create_commit([...CommitOperationAdd...])`,
    這裡抽成單一入口:同一次發佈的多個檔案,要嘛全進、要嘛全不進(原子性)。
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token or _hf_token())
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for dst, src in files]
    info = api.create_commit(repo_id=repo_id, repo_type=repo_type, revision=revision,
                             operations=ops, commit_message=message)
    return info.oid


# ── 發佈前健檢:抽樣開圖、統計尺寸、數壞檔/重複 ────────────────────────────
def verify_dataset(train_dir: str | Path, *, probe: int = 2000, seed: int = 1337) -> dict:
    """掃資料夾、隨機抽 probe 張開圖驗證,回傳 verify_report(發佈前的品質 gate)。

    壞檔(開不起來)、重複、尺寸分佈都記進報告,連同資料一起版本化上傳 →
    任何一版都查得到「當時的資料長怎樣」。
    """
    from PIL import Image
    train_dir = Path(train_dir)
    rel_paths = sorted(
        p.relative_to(train_dir).as_posix()
        for p in train_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)

    random.seed(seed)
    k = min(probe, len(rel_paths))
    sampled = random.sample(rel_paths, k) if k else []
    broken, sizes = 0, []
    for rel in sampled:
        try:
            with Image.open(train_dir / rel) as im:
                sizes.append(im.size)
        except Exception:
            broken += 1

    return {
        "n_files_total": len(rel_paths),
        "n_probed": k,
        "n_broken": broken,
        "duplicate_groups": len(rel_paths) - len(set(rel_paths)),
        "common_sizes_topN": Counter(sizes).most_common(15),
    }


# ── 發佈資料集:TAR + manifest(items.csv/paths.json)+ 驗證報告,一次 commit ──
def publish_dataset(repo_id: str, *, tar_path, items_csv, paths_json,
                    verify_report, token: str | None = None) -> str:
    """把資料集 bundle 版本化推上 HF Hub(對應 Route A 的 Push 段),回傳 commit oid。"""
    files = [
        ("archives/" + Path(tar_path).name, tar_path),
        ("items.csv", items_csv),
        ("paths.json", paths_json),
        ("verify_report.json", verify_report),
    ]
    try:
        n = len(json.loads(Path(paths_json).read_text(encoding="utf-8")))
    except Exception:
        n = "?"
    return commit_files(repo_id, files, f"dataset: TAR + metadata ({n} files)", token=token)


# ── 發佈模型 artifacts:向量/索引/keys/meta,date-stamp 版本化 + 更新 latest ──
def publish_artifacts(run_dir: str | Path, repo_id: str, *, model_tag: str = "vit-b-32",
                      version: str | None = None, token: str | None = None) -> dict:
    """把 run_dir 裡的 artifacts 推到 artifacts/<model_tag>/<version>/,並把 latest 指過去。

    version 預設用今天日期(date-stamp);回傳 {version, commit, files}。
    """
    run_dir = Path(run_dir)
    version = version or time.strftime("%Y-%m-%d")
    base = f"artifacts/{model_tag}/{version}"

    files = [(f"{base}/{p.name}", p)
             for name in _ARTIFACT_NAMES if (p := run_dir / name).exists()]
    if not files:
        raise FileNotFoundError(f"{run_dir} 找不到任何 artifacts({_ARTIFACT_NAMES})")

    oid = commit_files(repo_id, files, f"artifacts: {model_tag} {version}", token=token)
    update_latest(repo_id, version, model_tag=model_tag, token=token)
    return {"version": version, "commit": oid, "files": [dst for dst, _ in files]}


# ── latest 指標(重構新增):解耦「產出者」與「消費者」 ─────────────────────
def update_latest(repo_id: str, version: str, *, model_tag: str = "vit-b-32",
                  token: str | None = None) -> str:
    """寫 artifacts/<model_tag>/latest.json = {version},讓下游永遠只讀 latest,回傳 commit oid。

    比起「叫所有下游改路徑到新日期」,只更新一個指標檔就完成切版 —— 這正是 model
    registry 裡 `latest` / `production` 別名在做的事。
    (HF 原生也能用 git tag / branch 當別名;這裡用指標檔,最直觀好讀。)
    """
    pointer = {"version": version, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = Path(CFG.paths.run_dir) / f"_latest_{model_tag}.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    return commit_files(repo_id, [(f"artifacts/{model_tag}/latest.json", tmp)],
                        f"latest → {version}", token=token)


def resolve_latest(repo_id: str, *, model_tag: str = "vit-b-32",
                   token: str | None = None) -> str:
    """讀 latest.json,回傳目前 latest 指向的版本字串。"""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                           filename=f"artifacts/{model_tag}/latest.json",
                           token=token or _hf_token())
    return json.loads(Path(path).read_text(encoding="utf-8"))["version"]


def download_artifacts(repo_id: str, *, version: str = "latest", model_tag: str = "vit-b-32",
                       local_dir: str | Path | None = None, token: str | None = None) -> Path:
    """消費端入口:下載某版本(或 latest)的 artifacts 到本機,回傳該版本目錄。"""
    from huggingface_hub import snapshot_download
    if version == "latest":
        version = resolve_latest(repo_id, model_tag=model_tag, token=token)
    local_dir = Path(local_dir or (CFG.paths.run_dir / "hub_cache"))
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      allow_patterns=[f"artifacts/{model_tag}/{version}/*"],
                      local_dir=str(local_dir), token=token or _hf_token())
    return local_dir / "artifacts" / model_tag / version

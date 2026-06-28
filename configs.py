"""集中設定(single source of truth)。

【為什麼這支檔存在 — 面試可以這樣講】
原始 notebook 把參數(模型名、學習率、路徑、HF repo…)散落在各個 cell 的全域
變數裡,改一個值要翻好幾頁、還常常前後不一致。重構的第一步就是把「會變動的東西」
全部集中到一個地方:
  1. 一眼看完所有可調參數(沒有藏在程式深處的 magic number)。
  2. 路徑用環境變數 + 合理預設 → 同一份程式在 Colab / 本機 / 別人電腦都能跑。
  3. 機密(HF_TOKEN)只從環境變數讀,**絕不寫死進程式、絕不進 git**。
這就是「設定與程式碼分離(separation of configuration from code)」。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# 專案根目錄(本檔所在處),所有相對路徑都以它為基準
ROOT = Path(__file__).resolve().parent


def _path_env(var: str, default: Path) -> Path:
    """讀環境變數路徑;沒設就用預設。讓路徑可在不同機器覆寫。"""
    return Path(os.getenv(var, str(default)))


@dataclass(frozen=True)
class ModelConfig:
    name: str = "ViT-B-32"          # open_clip backbone
    pretrained: str = "openai"      # 預訓練權重來源
    # 把「文字端固定為 anchor、只訓練影像投影頭」是本專案的核心設計(見 README)
    train_image_head_only: bool = True


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 2             # 等效 batch = 32 * 2
    lr: float = 5e-5
    epochs: int = 3
    steps_per_epoch: int = 2000
    amp: bool = True                # 混合精度,省顯存、加速
    seed: int = 42


@dataclass(frozen=True)
class EvalConfig:
    max_samples: int = 1500         # R@K 評估抽樣數
    topk: int = 10                  # Top-K 推薦 / R@10


@dataclass(frozen=True)
class DataConfig:
    # WebDataset shards(.tar)目錄;預設本機 data/,可用環境變數覆寫成 Colab/Drive 路徑
    shards_dir: Path = field(default_factory=lambda: _path_env("AIF_SHARDS_DIR", ROOT / "data" / "shards"))
    shards_limit: int = 0           # 只用前 N 個 shard(0 = 全部)
    # Hugging Face 資料集(找不到本地 shards 時的保底來源)
    hf_repo_id: str = "Kai1023/kai-outfit-ai-dataset"
    hf_train_slice: str = "train[:10%]"
    hf_val_slice: str = "validation[:10%]"
    img_keys: tuple = ("jpg", "jpeg", "png", "webp", "bmp")
    txt_keys: tuple = ("txt", "caption", "caption.txt", "json")
    default_caption: str = "fashion outfit"


@dataclass(frozen=True)
class PathConfig:
    run_dir: Path = field(default_factory=lambda: _path_env("AIF_RUN_DIR", ROOT / "runs" / "v7"))
    thumb_side: int = 256

    @property
    def index_dir(self) -> Path:
        return self.run_dir / "index"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # 機密只從環境變數讀,絕不寫死(讀公開資料集可留空)
    @property
    def hf_token(self) -> str:
        return os.getenv("HF_TOKEN", "").strip()


# 全專案共用的單一設定實例
CFG = Config()

"""Caption:讓文字監督訊號「可學」—— 本專案檢索 lift 的真正來源。

來源:
  - ai_fashion_..._blip_v7.py Cell 3(_clean_words / _heuristic_caption /
    _load_cap_map / _extract 的 caption 解析段)。
  - 處理資料集part2.py 行 977–1002(Salesforce/blip-image-captioning-base 生成式 caption)。

──────────────────────────────────────────────────────────────────────────
因果鏈(面試一定要講清楚,呼應「為什麼 zero-shot 贏 finetuned」的 ablation)
──────────────────────────────────────────────────────────────────────────
啟發式 caption(資料夾名 + 檔名 + "outfit")很弱:短、重複、沒服飾細節 →
文字 anchor 模糊 → 檢索學不到語義。V7 的關鍵強化是用 **BLIP 生成式 caption** 把
弱字換成真正的描述,寫進 `cap_map.jsonl`,訓練時透過「覆寫鏈」蓋掉啟發式:

    BLIP caption → cap_map.jsonl → 蓋掉 heuristic → 文字 anchor 變強 → R@10 ↑

這就是把 R@10 從 0.009 拉到 ~0.7 的主因 —— **資料/caption 品質,不是 fine-tuning**。

caption 解析優先序(resolve_caption):
    1. WDS sample 自帶的 json.caption / json.text
    2. 純文字側 txt / caption 檔
    3. 檔名啟發式(heuristic_caption,保底)
    4. cap_map 覆寫(BLIP 產的高品質 caption 從這裡蓋上去)
    5. 噪聲過濾:tokens < min_tokens 視為無資訊 → 丟棄(回 None)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CAPTION_DEFAULT = "fashion outfit"
_TXT_KEYS = ("txt", "caption", "caption.txt", "json")
_SPLIT_FOLDERS = {"train", "val", "validation", "test"}


def clean_words(s: str) -> str:
    """把檔名/資料夾名清成空白分隔的可讀詞:去副檔名、底線/連字號轉空白、去雜符號。"""
    s = os.path.splitext(os.path.basename(str(s)))[0]
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def heuristic_caption(sample: dict) -> str:
    """沒有人工/模型 caption 時的保底:資料夾名 + 檔名關鍵字 + "outfit"。

    這是「很弱」的 baseline —— BLIP captioning 存在的理由就是要取代它(見模組 docstring)。
    """
    folder = os.path.basename(os.path.dirname(sample.get("__url__", "")))
    parts = []
    if folder and folder.lower() not in _SPLIT_FOLDERS:
        parts.append(clean_words(folder))
    if sample.get("__key__"):
        parts.append(clean_words(sample["__key__"]))
    parts.append("outfit")
    return " ".join(p for p in parts if p) or CAPTION_DEFAULT


def load_cap_map(path: str | Path) -> dict[str, str]:
    """讀 cap_map.jsonl → {key: caption}。每行一筆 {key/image_path/__key__, caption}。"""
    cap_map: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return cap_map
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = obj.get("key") or obj.get("image_path") or obj.get("__key__")
        cap = obj.get("caption")
        if key and cap:
            cap_map[str(key)] = str(cap)
    return cap_map


def resolve_caption(sample: dict, cap_map: dict[str, str] | None = None, *,
                    min_tokens: int = 3) -> str | None:
    """決定一個 WDS sample 該用哪段 caption(優先序鏈)。

    回傳 caption 字串;若過濾後判定為噪聲(tokens < min_tokens)則回 None,代表「丟棄此樣本」。
    影像解碼不在這裡(屬 webdataset 的職責),這支只負責「文字監督訊號」。
    """
    cap = None
    js = sample.get("json")
    if isinstance(js, dict):
        cap = js.get("caption") or js.get("text")
    if cap is None:
        for k in _TXT_KEYS:
            if k in sample and k != "json":
                v = sample[k]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", "ignore")
                cap = str(v)
                break
    if not cap or not cap.strip():
        cap = heuristic_caption(sample)

    # cap_map 覆寫:BLIP 產的高品質 caption 從這裡蓋掉啟發式
    if cap_map:
        key = sample.get("__key__", "")
        if key in cap_map:
            cap = cap_map[key]

    # 噪聲過濾:太短(沒資訊量)的 caption 丟掉,避免污染對比學習
    if len(re.findall(r"[a-z0-9]+", str(cap).lower())) < min_tokens:
        return None
    return cap


class BlipCaptioner:
    """BLIP 生成式 captioner(Salesforce/blip-image-captioning-base)。

    把「弱啟發式 caption」換成「真正描述性 caption」的引擎;產出寫進 cap_map.jsonl,
    訓練時經 resolve_caption 的覆寫鏈生效。模型較大(torch + transformers),故延遲載入。
    """

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base",
                 device: str | None = None, max_new_tokens: int = 30):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    def caption(self, images):
        """對單張或一批 PIL 影像生成 caption。傳單張回 str,傳 list 回 list[str]。"""
        import torch
        single = not isinstance(images, (list, tuple))
        batch = [images] if single else list(images)
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        caps = [c.strip() for c in self.processor.batch_decode(out, skip_special_tokens=True)]
        return caps[0] if single else caps


def build_cap_map(samples, captioner: BlipCaptioner, out_path: str | Path, *,
                  batch_size: int = 32, max_items: int = 0) -> int:
    """對一串 (key, PIL影像) 跑 BLIP captioning,寫成 cap_map.jsonl,回傳寫入筆數。

    samples:可迭代的 (key, pil);訓練 + 驗證影像都可餵進來一起建表。
    max_items > 0 時只處理前 N 張(快速試跑用)。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        keys_buf, imgs_buf = [], []

        def _flush():
            nonlocal written
            if not imgs_buf:
                return
            for key, cap in zip(keys_buf, captioner.caption(imgs_buf)):
                f.write(json.dumps({"key": key, "caption": cap}, ensure_ascii=False) + "\n")
                written += 1
            keys_buf.clear()
            imgs_buf.clear()

        for key, pil in samples:
            keys_buf.append(key)
            imgs_buf.append(pil)
            if len(imgs_buf) >= batch_size:
                _flush()
            if max_items and written >= max_items:
                break
        _flush()

    return written

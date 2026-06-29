"""重排序:先粗排(向量檢索)再精排,提升 Top-K 品質。

來源:處理資料集part2.py 行 851–895(BLIP-ITM 精排 + 兩階段 search_outfits)
     + ai_fashion_..._blip_v7.py Cell 9(屬性零樣本打分 + 迷你 KG 重排)。

──────────────────────────────────────────────────────────────────────────
為什麼要兩階段(retrieval → re-rank)—— 面試核心架構題
──────────────────────────────────────────────────────────────────────────
  第一段「粗排 / retrieve」:雙塔(two-tower)向量索引。query 與全庫各自編碼成向量、
    查 ANN。便宜、可擴展到百萬級,但圖與文「各編各的、沒有交互」→ 精度有上限。
  第二段「精排 / re-rank」:只對粗排取回的 ~60 個候選,用 BLIP-ITM 這種*交叉編碼器*
    (影像 + 文字一起進模型、直接算 match 分數)重排。準但貴(每候選一次前向),
    所以只在「少量候選」上跑才划算。
  → 這就是工業界檢索系統的標準形狀:recall 階段(快、廣)+ precision 階段(準、窄)。

  本檔提供的精排訊號:
    1. attribute_score   :零樣本屬性打分(袖長/外套…,CLIP 對 "a photo of X clothing")。
    2. BlipItmReranker   :BLIP-ITM 交叉編碼(主力精排)。
    3. 原 notebook 還疊了一層迷你 KG(同義詞正規化 + 互斥規則);屬專案特化,本精煉版
       以擴充點形式保留、未內建(見 notebooks/ 對照)。
"""
from __future__ import annotations

import numpy as np

# 預設零樣本屬性詞表(對齊 v7 Cell 9)
DEFAULT_ATTRIBUTES = {
    "sleeve_length": ["no-sleeve", "short-sleeve", "long-sleeve"],
    "outerwear": ["with jacket", "without jacket"],
}


def attribute_score(model, images, attributes: dict | None = None, *,
                    template: str = "a photo of {v} clothing") -> dict:
    """零樣本屬性打分:每個屬性群回傳 (候選值, 機率[N, V])。

    用 CLIP 把 "a photo of {value} clothing" 編成文字向量,與影像向量算 cosine 後 softmax。
    完全不需標註(zero-shot),可拿來當精排的補強訊號或做可解釋標籤。
    """
    import torch
    attributes = attributes or DEFAULT_ATTRIBUTES
    img = model.encode_image_features(images)               # (N, d) L2
    result = {}
    for group, values in attributes.items():
        txt = model.encode_text_anchor(model.tokenizer([template.format(v=v) for v in values]))
        probs = torch.softmax(img @ txt.t(), dim=-1).cpu().numpy()   # (N, V)
        result[group] = (values, probs)
    return result


class BlipItmReranker:
    """BLIP-ITM 交叉編碼重排器(Salesforce/blip-itm-base-coco)。

    交叉編碼器:影像 + 文字一起進模型,輸出「是否匹配」分數。比雙塔準,但每個候選都要
    一次前向,故只用在粗排取回的少量候選上。模型約 1.7G,延遲載入。
    """

    def __init__(self, model_id: str = "Salesforce/blip-itm-base-coco",
                 device: str | None = None, batch_size: int = 8):
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_id).to(self.device).eval()

    def score(self, images: list, text: str) -> np.ndarray:
        """對一批 PIL 影像與單一 text 算 ITM 匹配分數,回傳 np.ndarray[N]。"""
        import torch
        scores = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self.processor(images=list(batch), text=[text] * len(batch),
                                    return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                itm = self.model(**inputs).itm_score
            # 改進:itm_score 為 [N,2](match/no-match)時取「匹配」機率,比原 squeeze 穩健
            if itm.ndim == 2 and itm.shape[-1] == 2:
                itm = torch.softmax(itm, dim=-1)[:, 1]
            else:
                itm = itm.squeeze(-1)
            scores.append(itm.float().cpu().numpy())
        return np.concatenate(scores, 0) if scores else np.zeros(0, dtype="float32")

    def rerank(self, images: list, text: str, k: int):
        """回傳 (order, scores):order 為依 ITM 分數由高到低的前 k 個候選索引。"""
        scores = self.score(images, text)
        return np.argsort(-scores)[:k], scores


def build_query_vector(model, positive, negative=None, *, neg_weight: float = 0.4) -> np.ndarray:
    """把正/負文字 prompt 編成單一 query 向量(L2 normalized),回傳 np.ndarray[1, d]。

    多個正樣本取平均;有負樣本時做 pos − neg_weight·neg(把不想要的語義方向推開)——
    典型的「正減負」query 技巧。
    """
    import torch.nn.functional as F
    if isinstance(positive, str):
        positive = [positive]
    pos = model.encode_text_anchor(model.tokenizer(positive)).mean(0, keepdim=True)
    if negative:
        if isinstance(negative, str):
            negative = [negative]
        neg = model.encode_text_anchor(model.tokenizer(negative)).mean(0, keepdim=True)
        q = F.normalize(pos - neg_weight * neg, dim=-1)
    else:
        q = F.normalize(pos, dim=-1)
    return q.cpu().numpy().astype("float32")


def two_stage_search(model, index, keys, positive, *, negative=None, k: int = 12,
                     preselect: int = 60, reranker: BlipItmReranker | None = None,
                     fetch_image=None) -> list[tuple]:
    """兩階段檢索:向量粗排取回 preselect 個候選 →(可選)BLIP-ITM 精排 → 回前 k。

    回傳 list[(key, score)]。reranker=None 時只做粗排(回向量相似度);
    有 reranker 時須提供 fetch_image(key)->PIL 以載候選圖。
    """
    from .index import search
    qv = build_query_vector(model, positive, negative)
    cand_idx, sims = search(index, qv, k=min(preselect, len(keys)))
    cand_idx, sims = list(cand_idx[0]), list(sims[0])

    if reranker is None:
        return [(keys[i], float(s)) for i, s in zip(cand_idx, sims)][:k]

    if fetch_image is None:
        raise ValueError("使用 reranker 時須提供 fetch_image(key)->PIL")
    text = positive[0] if isinstance(positive, (list, tuple)) else positive
    pil_list = [fetch_image(keys[i]) for i in cand_idx]
    order, itm_scores = reranker.rerank(pil_list, text, k)
    return [(keys[cand_idx[i]], float(itm_scores[i])) for i in order][:k]

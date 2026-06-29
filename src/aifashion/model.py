"""模型:CLIP backbone + 影像投影頭(anchor-to-T0)。

來源:ai_fashion_..._blip_v7.py Cell 5(open_clip 載入、encode_images/encode_text)
     + Cell 6.5A(_eye_init、img_proj/txt_proj、logit_scale)。

──────────────────────────────────────────────────────────────────────────
核心設計(面試第一個主打 — 怎麼修掉 embedding 空間漂移 space drift)
──────────────────────────────────────────────────────────────────────────
V5/V6 同時微調影像端與文字端,兩邊座標一起動 → 互相遷就、漂移,檢索學不到
穩定語義(R@10 掉到 0.009)。V7 的修法:

  1. 凍結 CLIP backbone(只當特徵抽取器,不動)。
  2. 把「文字端固定成 anchor」:txt_proj 凍結成 Identity → 文字向量恆等於
     backbone 的 T0,當作不動的語義座標。
  3. 只訓練「影像投影頭」img_proj,讓影像向量 I1 去對齊 T0。
  4. 兩個頭都以 Identity 初始化 → 訓練從「I1 ≈ I0」這個乾淨起點出發,
     不會繼承到先前的壞權重。

效率上的好處:backbone 凍結 → 它的特徵用 no_grad 算就好,梯度只流經那一層
小小的 img_proj,訓練很便宜。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_identity_(linear: nn.Linear) -> None:
    """把線性層權重設成(左上角)單位矩陣;當投影頭的 no-op 起點。"""
    with torch.no_grad():
        w = linear.weight
        w.zero_()
        n = min(w.shape[0], w.shape[1])
        w[:n, :n] = torch.eye(n, device=w.device, dtype=w.dtype)


class AnchoredCLIP(nn.Module):
    """凍結的 CLIP backbone + 可訓練影像頭 + 凍結的文字 anchor 頭。

    用法:
        m = AnchoredCLIP.from_pretrained(device="cuda")
        i0 = m.encode_image_features(images)   # 凍結 backbone 的影像向量
        t0 = m.encode_text_anchor(tokens)      # 文字 anchor(不動)
        i1 = m.project_image(i0)               # 訓練中的影像向量(梯度只到這)
    """

    def __init__(self, backbone: nn.Module, tokenizer, preprocess_train,
                 preprocess_eval, embed_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_eval = preprocess_eval

        # 兩個投影頭:img_proj 可訓練、txt_proj 當凍結的 anchor(見模組 docstring)
        self.img_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.txt_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # 溫度(對比相似度的縮放);log 空間存,用時 exp
        self.logit_scale = nn.Parameter(torch.tensor(0.07).log())

        self._freeze(self.backbone)          # backbone 不訓練
        self.reset_heads_to_identity()       # 兩頭都從 Identity 起步
        self._freeze(self.txt_proj)          # 文字端錨定:凍結 txt_proj

    # ---- 建構 ----
    @classmethod
    def from_pretrained(cls, model_name: str = "ViT-B-32",
                        pretrained: str = "openai", device: str | None = None):
        """載入 open_clip backbone + transforms + tokenizer 並組裝。

        預設值對齊 configs.ModelConfig;呼叫端通常傳 CFG.model.name / .pretrained。
        """
        import open_clip
        backbone, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        embed_dim = int(getattr(backbone.visual, "output_dim", 512))
        model = cls(backbone, tokenizer, preprocess_train, preprocess_eval, embed_dim)
        return model.to(device) if device else model

    # ---- 凍結 / 初始化 ----
    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    def reset_heads_to_identity(self) -> None:
        """把兩個投影頭重設成 Identity(避免載回先前的壞權重)。"""
        _init_identity_(self.img_proj)
        _init_identity_(self.txt_proj)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ---- 編碼 ----
    @torch.no_grad()
    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """凍結 backbone 的影像向量 I0(L2 normalized)。"""
        feats = self.backbone.encode_image(images.to(self.device))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text_anchor(self, tokens: torch.Tensor) -> torch.Tensor:
        """文字 anchor T0(L2 normalized)。txt_proj 是凍結 Identity,故 T0 即 backbone 文字向量。"""
        feats = self.backbone.encode_text(tokens.to(self.device))
        return F.normalize(self.txt_proj(feats), dim=-1)

    def project_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """把影像向量過(可訓練的)img_proj → I1(L2 normalized);梯度只流經這裡。"""
        return F.normalize(self.img_proj(image_features.to(self.device)), dim=-1)

    @property
    def trainable_parameters(self):
        """要丟給 optimizer 的參數:只有影像投影頭。"""
        return self.img_proj.parameters()

    # ---- 存 / 載投影頭(與原始 last_extra.pt 格式相容)----
    def save_heads(self, path: str | Path) -> None:
        torch.save({
            "img_proj": self.img_proj.state_dict(),
            "txt_proj": self.txt_proj.state_dict(),
            "logit_scale": self.logit_scale.detach().float().cpu(),
        }, str(path))

    def load_heads(self, path: str | Path, map_location: str | None = None) -> None:
        ckpt = torch.load(str(path), map_location=map_location or self.device)
        self.img_proj.load_state_dict(ckpt["img_proj"])
        self.txt_proj.load_state_dict(ckpt["txt_proj"])
        with torch.no_grad():
            self.logit_scale.copy_(ckpt["logit_scale"].to(self.logit_scale.device))

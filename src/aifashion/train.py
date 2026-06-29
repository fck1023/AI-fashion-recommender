"""訓練:只訓影像投影頭,把影像向量對齊文字 anchor T0。

來源:ai_fashion_..._blip_v7.py Cell 6.5A-reset(實際執行的那版;Cell 6 / 6.5 / 6.5A
     是被註解掉的歷史實驗版,不搬)。

──────────────────────────────────────────────────────────────────────────
對比損失:為什麼是「單向 in-batch InfoNCE」(面試核心題)
──────────────────────────────────────────────────────────────────────────
一個 batch 有 B 對(影像 Iᵢ、它的文字 anchor Tᵢ)。把 B 張影像向量和 B 個文字
anchor 兩兩內積 → B×B 的相似度矩陣 logits。對第 i 列來說:
  - 對角線 logits[i,i] = Iᵢ 對自己的 Tᵢ(**正樣本**,要拉高)。
  - 同列其他 B−1 個 = Iᵢ 對「batch 內別人的文字」(**負樣本**,要壓低)。
於是「把第 i 列分類成第 i 類」這個 cross_entropy,就等於 InfoNCE:正樣本當分子、
整列當分母。batch 裡其他樣本自動充當負樣本(in-batch negatives),不必另外挖負例。

為什麼**單向**(只 image→text,不像 CLIP 原版做雙向)?
因為文字端被凍結成 anchor、根本不動 —— 只有影像需要往固定的 anchor 靠。
雙向是「兩端都在學」時才需要;這裡只訓一端,單向就夠,而且更穩。

logit_scale(溫度)用 exp 還原並 clamp 在 [1e-3, 100]:控制 softmax 的銳利度。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .configs import CFG
from .eval import report_zero_shot_vs_finetuned


def info_nce_anchor_loss(image_emb: torch.Tensor, text_anchor: torch.Tensor,
                         logit_scale: torch.Tensor) -> torch.Tensor:
    """單向 in-batch 對比損失:每張影像拉近自己的文字 anchor、推遠 batch 內其他 anchor。

    image_emb / text_anchor 都已 L2 normalize,shape (B, d)。回傳純量 loss。
    """
    logits = logit_scale * (image_emb @ text_anchor.t())    # (B, B);對角線為正樣本
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def train_image_head(model, train_loader, val_loader, *,
                     lr: float | None = None, steps: int | None = None,
                     grad_accum: int | None = None, max_grad_norm: float = 1.0,
                     amp: bool | None = None, save_path=None):
    """訓練影像投影頭(其餘全凍結),存檔並回傳權重路徑。

    流程:訓前 R@K 報告(確認從乾淨起點出發)→ AMP 訓練迴圈 → 存頭 → 訓後 R@K 報告。
    超參預設取自 configs.TrainConfig,可用關鍵字逐一覆寫。
    """
    cfg = CFG.train
    lr = cfg.lr if lr is None else lr
    steps = cfg.steps_per_epoch if steps is None else steps
    grad_accum = cfg.grad_accum if grad_accum is None else grad_accum
    device = model.device
    amp_enabled = (cfg.amp if amp is None else amp) and device.type == "cuda"

    # backbone 整個凍結維持 eval;只有影像頭進入 train 模式
    model.eval()
    model.img_proj.train()

    # optimizer 只拿可訓練的影像頭參數(weight_decay=0:單一線性投影不需要)
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # 訓前:I1–T0 應 ≈ I0–T0(Identity 起點),兩列幾乎相等才正常
    report_zero_shot_vs_finetuned(model, val_loader, CFG.eval.max_samples, tag="Before")

    optimizer.zero_grad(set_to_none=True)
    step = 0
    pbar = tqdm(total=steps, desc="train img head → anchor T0")
    while step < steps:
        for batch in train_loader:
            images, tokens = batch[0], batch[1]
            # backbone 特徵用 no_grad 算(凍結),梯度只會流經底下的 img_proj
            i0 = model.encode_image_features(images)
            t0 = model.encode_text_anchor(tokens)
            with torch.autocast(device.type, enabled=amp_enabled):
                i1 = model.project_image(i0)                # 可訓練影像頭
                scale = model.logit_scale.exp().clamp(1e-3, 100)
                loss = info_nce_anchor_loss(i1, t0, scale) / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                # ★ 修正:AMP 下先 unscale 再 clip。原 notebook Cell 6.5A-reset 漏了這步,
                #    導致在「被放大的梯度」上做 clip_grad_norm,等效門檻被偷偷放大、clip 失效。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.img_proj.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.set_postfix(loss=float(loss.item() * grad_accum))
            pbar.update(1)
            if step >= steps:
                break
    pbar.close()

    # 存投影頭(格式與原始 last_extra.pt 相容)
    save_path = Path(save_path or (CFG.paths.run_dir / "last_extra.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_heads(save_path)
    print(f"✅ 已存影像頭:{save_path}")

    # 訓後:finetuned 那列理想上 ≥ zero-shot
    model.img_proj.eval()
    report_zero_shot_vs_finetuned(model, val_loader, CFG.eval.max_samples, tag="After")
    return save_path

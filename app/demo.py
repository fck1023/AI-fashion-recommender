"""Gradio 互動式 demo:文字 / 圖片查詢 → Top-K 穿搭推薦。

來源:驗證一下master_kai_的成果.py 行 1142–1234(faiss_search / make_contact_sheet /
     兩分頁 Gradio UI)+ ai_fashion_..._blip_v7.py Cell 9 的 UI。

整條 pipeline 的「消費端」:載入訓練好的模型 + index artifacts,把使用者查詢轉成
Top-K 穿搭。artifacts 可從本地 run_dir(index.load_artifacts)或 HF Hub 的 `latest`
別名(hub.download_artifacts)取回 —— 對應 README 的雙雲發佈契約。

執行:
    pip install -e .              # 先讓 aifashion 套件可被 import
    python app/demo.py            # 預設讀本地 runs/v7 的 artifacts
"""
from __future__ import annotations

from pathlib import Path

from aifashion.configs import CFG
from aifashion.index import load_artifacts, search
from aifashion.model import AnchoredCLIP


class RetrievalEngine:
    """載入模型 + 索引 artifacts,提供文字 / 圖片檢索(demo 與 API 共用)。"""

    def __init__(self, *, run_dir=None, images_dir=None, repo_id: str | None = None,
                 device: str | None = None):
        self.model = AnchoredCLIP.from_pretrained(CFG.model.name, CFG.model.pretrained, device=device)
        if run_dir is None and repo_id:                 # 從 HF Hub latest 取回 artifacts
            from aifashion.data.hub import download_artifacts
            run_dir = download_artifacts(repo_id)
        self.index, self.keys, self.meta = load_artifacts(run_dir or CFG.paths.run_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.repo_id = repo_id

    # ── 檢索 ──────────────────────────────────────────────────────────────
    def search_text(self, text: str, k: int = 12):
        qv = self.model.encode_text_anchor(self.model.tokenizer([text])).cpu().numpy().astype("float32")
        idx, sims = search(self.index, qv, k)
        return [(str(self.keys[i]), float(s)) for i, s in zip(idx[0], sims[0])]

    def search_image(self, pil, k: int = 12):
        x = self.model.preprocess_eval(pil).unsqueeze(0)
        v = self.model.encode_image_features(x).cpu().numpy().astype("float32")
        idx, sims = search(self.index, v, k)
        return [(str(self.keys[i]), float(s)) for i, s in zip(idx[0], sims[0])]

    # ── 取圖:本地 images_dir 優先,否則回退 HF dataset ────────────────────
    def fetch_image(self, key: str):
        from PIL import Image
        if self.images_dir:
            p = self.images_dir / key
            if p.exists():
                return Image.open(p).convert("RGB")
        if self.repo_id:
            from huggingface_hub import hf_hub_download
            fp = hf_hub_download(self.repo_id, filename=f"raw/train/{key}",
                                 repo_type="dataset", token=CFG.hf_token or None)
            return Image.open(fp).convert("RGB")
        return None


def make_contact_sheet(engine: RetrievalEngine, results, *, cols: int = 6, thumb: int = 256):
    """把 Top-K 結果貼成棋盤縮圖,左上角標相似度;取不到圖就放灰格。"""
    from PIL import Image, ImageDraw
    cells = []
    for key, score in results:
        try:
            im = engine.fetch_image(key).convert("RGB").resize((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (40, 40, 40))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 80, 28), fill=(0, 0, 0))
        draw.text((6, 8), f"{score:.2f}", fill=(255, 255, 255))
        cells.append(im)
    if not cells:
        return Image.new("RGB", (thumb, thumb), (0, 0, 0))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (20, 20, 20))
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * thumb, r * thumb))
    return sheet


def _format_ranking(results) -> str:
    return "\n".join(f"{i + 1}. {key}  ({score:.3f})" for i, (key, score) in enumerate(results))


def build_demo(engine: RetrievalEngine | None = None, **engine_kwargs):
    """組裝並回傳 Gradio app(尚未啟動)。engine 為 None 時依 engine_kwargs 建立。"""
    import gradio as gr
    engine = engine or RetrievalEngine(**engine_kwargs)

    def _text_ui(query, topk):
        if not query or not query.strip():
            return None, "請輸入文字"
        res = engine.search_text(query.strip(), int(topk))
        return make_contact_sheet(engine, res), _format_ranking(res)

    def _image_ui(image, topk):
        if image is None:
            return None, "請上傳圖片"
        from PIL import Image
        res = engine.search_image(Image.fromarray(image), int(topk))
        return make_contact_sheet(engine, res), _format_ranking(res)

    with gr.Blocks() as demo:
        gr.Markdown("## 穿搭檢索 Demo(CLIP ViT-B-32 + FAISS)")
        with gr.Tab("Text → Image"):
            q = gr.Textbox(label="文字查詢", value="black leather jacket")
            topk = gr.Slider(3, 24, value=12, step=1, label="Top-K")
            out_img = gr.Image(type="pil", label="結果縮圖拼貼")
            out_txt = gr.Textbox(label="排名與分數", lines=8)
            gr.Button("Search").click(_text_ui, [q, topk], [out_img, out_txt])
        with gr.Tab("Image → Image"):
            img = gr.Image(type="numpy", label="上傳查詢圖片")
            topk2 = gr.Slider(3, 24, value=12, step=1, label="Top-K")
            out_img2 = gr.Image(type="pil", label="結果縮圖拼貼")
            out_txt2 = gr.Textbox(label="排名與分數", lines=8)
            gr.Button("Search").click(_image_ui, [img, topk2], [out_img2, out_txt2])

    return demo


if __name__ == "__main__":
    build_demo().launch()

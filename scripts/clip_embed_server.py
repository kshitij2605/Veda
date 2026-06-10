"""Minimal OpenAI-compatible CLIP embedding service for Veda's multimodal path.

Veda's veda-core embedding client posts to `{endpoint}/v1/embeddings` with either:
  - text:  {"model": ..., "input": "<text>"}
  - image: {"model": ..., "messages": [{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
              {"type":"text","text":"Represent the given image."}]}]}
and parses {"data":[{"embedding":[...]}]}.

CLIP ViT-B/32 projects text + images into one 512-d space, so text queries match
images. Runs on a single GPU; model is read from the local HF cache.
"""
import base64
import io
import re

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

app = FastAPI()
_DATA_URI = re.compile(r"^data:[^;]+;base64,(.*)$", re.S)


@torch.no_grad()
def embed_text(texts):
    # transformers 5.x's get_text_features returns an output object, not a
    # tensor — use the sub-model + projection directly for the shared-space embedding.
    inp = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = model.text_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
    feats = model.text_projection(out.pooler_output)
    return F.normalize(feats, p=2, dim=-1).cpu().tolist()


@torch.no_grad()
def embed_image(img):
    inp = processor(images=img, return_tensors="pt").to(DEVICE)
    out = model.vision_model(pixel_values=inp["pixel_values"])
    feats = model.visual_projection(out.pooler_output)
    return F.normalize(feats, p=2, dim=-1).cpu().tolist()


def decode_image(url):
    m = _DATA_URI.match(url)
    raw = base64.b64decode(m.group(1) if m else url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dimension": DIM, "device": DEVICE}


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    vectors = []

    if "input" in body:
        inp = body["input"]
        texts = inp if isinstance(inp, list) else [inp]
        vectors = embed_text(texts)
    elif "messages" in body:
        imgs = []
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        imgs.append(decode_image(part["image_url"]["url"]))
        if imgs:
            for img in imgs:
                vectors.extend(embed_image(img))
        else:
            txt = " ".join(
                p.get("text", "")
                for msg in body["messages"]
                for p in (msg["content"] if isinstance(msg.get("content"), list) else [{"text": msg.get("content", "")}])
            )
            vectors = embed_text([txt or " "])
    else:
        return {"error": "request must include 'input' or 'messages'"}

    data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {
        "object": "list",
        "data": data,
        "model": body.get("model", MODEL_ID),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)

"""Omni embedder sidecar for Veda E2E testing.

Wraps jinaai/jina-embeddings-v5-omni-small behind an OpenAI-style
/v1/embeddings endpoint that accepts BOTH shapes Veda's embedding
client emits:

  1. Text:  {"input": "<text>", "input_type": "query"|"document"}
  2. Media (chat-style): {"messages": [{"role": "user",
       "content": [<media_part>, {"type": "text", "text": "<instr>"}]}]}
     media_part = image_url | audio_url | video_url with a
     data:<mime>;base64,<B64> URL.

Returns {"data": [{"embedding": [...1024 floats...]}],
         "model": "jina-v5-omni-small"}.
"""

import base64
import binascii
import tempfile
import traceback
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jinaai/jina-embeddings-v5-omni-small"
RETURN_MODEL = "jina-v5-omni-small"

MIME_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}

app = FastAPI()
model: Optional[SentenceTransformer] = None


@app.on_event("startup")
def _load() -> None:
    global model
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        model_kwargs={"default_task": "retrieval"},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": model is not None}


class EmbedRequest(BaseModel):
    model: Optional[str] = None
    input: Optional[Any] = None
    input_type: Optional[str] = None
    messages: Optional[list] = None
    dimensions: Optional[int] = None


def _decode_data_uri(url: str) -> str:
    """data:<mime>;base64,<B64>  ->  path to a NamedTemporaryFile."""
    if not url.startswith("data:"):
        raise ValueError(f"expected data: URI, got {url[:32]!r}")
    header, _, b64 = url.partition(",")
    mime = header[len("data:"):].split(";", 1)[0].strip().lower()
    suffix = MIME_SUFFIX.get(mime, ".bin")
    try:
        raw = base64.b64decode(b64, validate=False)
    except binascii.Error as exc:
        raise ValueError(f"bad base64: {exc}") from exc
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def _extract_media_url(part: dict) -> str:
    t = part.get("type", "")
    for key in ("image_url", "audio_url", "video_url"):
        if key in part:
            v = part[key]
            if isinstance(v, dict):
                return v.get("url", "")
            return v
    raise ValueError(f"no media url in part of type {t!r}")


def _encode_kwargs(req: EmbedRequest) -> dict:
    kw: dict = {}
    if req.dimensions:
        kw["truncate_dim"] = int(req.dimensions)
    return kw


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> JSONResponse:
    assert model is not None, "model not loaded"
    kw = _encode_kwargs(req)
    try:
        # --- Media (chat-style) shape ---
        if req.messages:
            media_path = None
            for msg in req.messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if any(k in part for k in
                           ("image_url", "audio_url", "video_url")):
                        media_path = _decode_data_uri(
                            _extract_media_url(part))
                        break
                if media_path:
                    break
            if media_path is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "no media part found in messages"},
                )
            vec = model.encode(media_path, **kw)
        # --- Text shape ---
        elif req.input is not None:
            text = req.input
            if isinstance(text, list):
                text = text[0] if text else ""
            it = (req.input_type or "document").lower()
            if it == "query":
                vec = model.encode_query(text, **kw)
            else:
                vec = model.encode_document(text, **kw)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "request has neither 'input' nor "
                                  "'messages'"},
            )
        emb = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        return JSONResponse(content={
            "data": [{"embedding": emb, "index": 0,
                      "object": "embedding"}],
            "model": RETURN_MODEL,
            "object": "list",
        })
    except Exception as exc:  # noqa: BLE001 - surface full traceback
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": tb},
        )

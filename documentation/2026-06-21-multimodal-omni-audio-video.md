# Unified Multimodal Embedding: jina-v5-omni, audio + video `/ingest-media` (2026-06-21)

This guide covers Veda's **unified multimodal embedding** path: ONE omni model that embeds text, image, audio, and video into a single 1024-d cross-modal vector space, plus the `/v1/collections/{c}/ingest-media` route and the audio/video arms of `/v1/embed`. It is the result of the multimodal-omni E2E campaign (branch `feat/multimodal-omni-e2e`).

## 1. The jina-v5-omni sidecar

`jina-embeddings-v5-omni-small` is a single embedding model that accepts **text, image, audio, and video** and projects all of them into the **same 1024-d cosine space** — so a text query can retrieve audio/video/image records and vice versa, with no per-modality model juggling. It runs as a sidecar (an OpenAI-style multimodal embedding server) at `http://localhost:8001`.

- Sidecar app: `tools/omni-embed/app.py` (accepts `image_url` / `audio_url` / `video_url` media parts; maps common MIME types: jpg/png/webp/gif, wav/mp3/flac/ogg, mp4/webm/mov).
- Veda preset: `feat(embedding) add jina-v5-omni-small multimodal preset` (`c30939f`) registers the model with modalities `[Text, Image, Audio, Video]`. The audio arm of the multimodal payload builder landed in `250431a`.
- GPU footprint: ~4.6 GB on one GPU. The first audio inference is a one-time ~60 s warmup; steady-state RTT is then ~0.5 s (text), ~0.9 s (audio), ~1.3 s (video), ~2.8 s (image). See `reports/2026-06-21-e2e-performance.md`.

## 2. Configuration — one model behind both EMBEDDING_* and MULTIMODAL_*

The unified-space behavior comes from pointing **both** the text and the multimodal embedding config at the same omni sidecar:

```sh
export EMBEDDING_ENDPOINT=http://localhost:8001
export MULTIMODAL_EMBEDDING_ENDPOINT=http://localhost:8001
# (model name resolves to the jina-v5-omni-small preset)
```

When both point at the omni model, text and media vectors are directly comparable (same encoder, same space). The model registers once with modalities `[Text, Image, Audio, Video]`; `resolve_modality` (`7cf989d`) routes each request to the right arm.

**Config gotcha (Finding 1 in the campaign bug log):** the `.env` loader used to clobber already-exported env vars; it now only sets keys that are *absent* (fix `ea3292a`). A stale `EMBEDDING_ENDPOINT` pointing at a text-only TEI server will make media payloads 422 ("missing field `input`") because the text endpoint can't serve `{messages}` media parts. Keep `EMBEDDING_*` and `MULTIMODAL_*` pointed at the omni sidecar.

## 3. `/v1/embed` — audio & video

`POST /v1/embed` accepts one of `text` / `image_b64` / `audio_b64` / `video_b64` (raw base64, no data-URI prefix needed):

```sh
# audio
curl -X POST localhost:6340/v1/embed -H 'content-type: application/json' \
  -d "{\"audio_b64\":\"$(base64 -w0 clip.wav)\"}"
# video
curl -X POST localhost:6340/v1/embed -H 'content-type: application/json' \
  -d "{\"video_b64\":\"$(base64 -w0 clip.mp4)\"}"
```

Response: `{model, modality, dimension:1024, values:[…1024 floats…], elapsed_ms}` (the vector key is **`values`**). `resolve_modality` (`7cf989d`) picks the modality from which field is present; the omni preset resolves image/audio/video without hijacking the text default (`4ee3c94`).

## 4. `/v1/collections/{c}/ingest-media` — upload-and-embed for media

`POST /v1/collections/{c}/ingest-media` (`989e79b`) is a multipart route: upload a media file and Veda infers the modality from the MIME type, embeds the bytes with the matching model, stores the raw bytes as a blob, and inserts a record — in one call.

**Collection shape required:** a vector field with `auto_embed { source: <blob field>, modality: image|audio|video }`, plus a `media_blob` Blob field (for the raw bytes), and optionally `source_path` / `class` string fields. Each `auto_embed` vector field is pinned to ONE modality, so a media collection has one vector field per modality you want to ingest.

```sh
curl -X POST localhost:6340/v1/collections/sounds/ingest-media \
  -F 'file=@dog_bark.wav' -F 'class=animal'
# to {status:ingested, records_inserted:1, vector_dim:1024, modality:Audio, blob_key:"…"}
```

Multipart parts: `file` (required), `class` / `source_path` (optional string fields). Returns `NO_MEDIA_FIELD` (422) if the collection has no `auto_embed` vector field matching the uploaded file's modality.

## 5. Unified cross-modal space — what works

With one omni model, you embed a **text** query and search a collection holding **audio/video/image** vectors directly (`POST /search { query:{ vector:{ field, values }}}`).

- **text↔audio: strong.** Validated on ESC-50: "a dog barking" surfaces the dog-bark audio (cosine dist ~0.50) above rooster audio (~0.69); "rain storm sound" surfaces rain clips. Ranking matches the ground-truth classes.
- **text↔text: strong.** Standard dense retrieval quality.
- **text→image: neutral** on generic document pages (no strong semantic signal expected).
- **text→video: weak (known caveat — see below).**

## 6. Known caveats

### text→video alignment is weak
With jina-v5-omni-small, video↔text cosine distances cluster ~0.78–0.89 with little content discrimination, so a matching text query does **not** reliably surface the matching video above unrelated media. The full ingest→embed→store→search pipeline works for video (vectors are produced and stored); the weakness is **semantic alignment quality**, likely from frame sampling / short-clip encoding, not from Veda. Mitigations to try: more frames, longer clips, a different video instruction, and a video↔video (intra-modal) retrieval check to confirm whether video vectors are discriminative at all.

### Index type is Flat (brute-force) on small data
The declared ANN index (HNSW/IVF-PQ/RaBitQ/SQ/…) only builds on **sealed** segments, and the seal threshold is hardcoded 512 MB with no seal/flush REST route. So small media collections stay in the growing segment and are brute-forced as **Flat** regardless of declared type — correct results, but no ANN speedup or recall/latency comparison through the live API. Tracked as a P1 follow-up: add a `POST /collections/{c}/seal` route and/or a `VEDA_SEAL_THRESHOLD` env. (Per-index algorithm correctness is unit-tested in `veda-index`.)

## See also
- `reports/2026-06-21-e2e-report.md` — full campaign report (feature matrix, bugs, suggestions).
- `reports/2026-06-21-e2e-performance.md` — measured per-modality embed RTT, ingest throughput, search/rerank/LLM latency.
- `reports/2026-06-21-e2e-bug-findings.md` — detailed bug log with repros and commit hashes.

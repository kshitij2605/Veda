# Multimodal & Cross-Modal Search

Veda is multimodal-native: text, images, audio, video, and documents are first-class, not bolted on. The sharpest expression of that is a **unified omni embedder** — one model that projects all modalities into a *single* vector space, so a text query can retrieve audio, video, and image records directly (and vice versa).

## The unified omni embedder

`jina-embeddings-v5-omni-small` is a single model that accepts **text, image, audio, and video** and projects all of them into the **same 1024-d cosine space**. With one encoder behind everything, there's no per-modality model juggling and — crucially — vectors from different modalities are **directly comparable**, which is what makes true cross-modal retrieval work.

You opt into the unified space by pointing **both** the text and the multimodal embedding config at the same omni sidecar:

```sh
export EMBEDDING_ENDPOINT=http://localhost:8001
export MULTIMODAL_EMBEDDING_ENDPOINT=http://localhost:8001
# the model name resolves to the jina-v5-omni-small preset (modalities: text, image, audio, video)
```

The omni model runs as an OpenAI-style embedding sidecar and needs a GPU (~4.6 GB). It is **optional** — the default stack still pairs a CPU text embedder (BGE-M3) with a separate multimodal model (Qwen3-VL); the omni model is the path to a single cross-modal space. See [embedding models](/docs/installation#embedding-models) for the config knobs.

> **Config gotcha.** Keep `EMBEDDING_*` and `MULTIMODAL_*` both pointed at the omni sidecar. A stale `EMBEDDING_ENDPOINT` pointing at a text-only server makes media payloads fail (the text endpoint can't serve media parts).

## Embedding media directly

`POST /v1/embed` accepts one of `text`, `image_b64`, `audio_b64`, or `video_b64` (raw base64) and returns the vector for that modality:

```sh
# embed an audio clip
curl -X POST localhost:6340/v1/embed -H 'content-type: application/json' \
  -d "{\"audio_b64\":\"$(base64 -w0 clip.wav)\"}"
```

The response is `{ model, modality, dimension: 1024, values: [ … ], elapsed_ms }` — Veda picks the modality from whichever field is present.

## Upload-and-embed: `/ingest-media`

`POST /v1/collections/{name}/ingest-media` is a multipart route that takes a media file and, in one call, **infers the modality from the MIME type, embeds the bytes, stores the raw bytes as a blob, and inserts a record**:

```sh
curl -X POST localhost:6340/v1/collections/sounds/ingest-media \
  -F 'file=@dog_bark.wav' -F 'class=animal'
# → { status: "ingested", records_inserted: 1, vector_dim: 1024, modality: "Audio", blob_key: "…" }
```

The collection needs a vector field with `auto_embed { source: <blob field>, modality: image|audio|video }`, a `media_blob` blob field for the raw bytes, and optionally `source_path` / `class` string fields. Each `auto_embed` vector field is pinned to **one** modality, so a media collection has one vector field per modality you want to ingest. A file whose modality has no matching `auto_embed` field returns `NO_MEDIA_FIELD` (`422`).

## Cross-modal search

With one omni model, you embed a **text** query and [search](/docs/search) a collection holding **audio / video / image** vectors directly — no separate query encoder, because they share the space:

```json
POST /v1/collections/sounds/search
{ "query": { "vector": { "field": "embedding", "values": [/* text-embedded query */] } }, "limit": 5 }
```

Validated cross-modal behavior on real data:

| Query → target | Quality |
|---|---|
| text → audio | **Strong** — "a dog barking" ranks the dog-bark clip above rooster/rain (ground-truth classes match) |
| text → text | **Strong** — standard dense retrieval |
| text → image | Neutral on generic document pages (no strong semantic signal expected) |
| text → video | **Weak** *(known caveat — below)* |

## Current limitations

- **text → video alignment is weak.** With this omni model, video↔text cosine distances cluster tightly with little content discrimination, so a matching text query does not reliably surface the matching video. The full ingest → embed → store → search path *works* for video (vectors are produced and stored); the weakness is semantic alignment quality (likely frame sampling / short-clip encoding), not the pipeline. More frames, longer clips, or an intra-modal (video → video) check are the mitigations to try.
- **The omni model needs a GPU** (~4.6 GB), with a one-time ~60 s warmup on the first audio inference. Text-only deployments can stay on the CPU BGE-M3 text embedder.

# Multimodal & Cross-Modal Search

Veda is multimodal-native: text, images, audio, video, and documents are first-class. The new capability here is that **audio and video are now fully wired through the pipeline** — embed them directly, upload-and-embed them into a collection, and retrieve them cross-modally — so they're no longer second-class next to text and image. Veda adds the audio/video arms of `/v1/embed`, modality-aware request routing, and the `/ingest-media` upload route; the embedding vectors themselves are produced by whatever model you point Veda at.

Cross-modal retrieval (a text query returning audio/video/image records) works best with a **unified embedding model** — one model that maps every modality into the *same* vector space, so vectors from different modalities are directly comparable. Veda ships a preset for `jina-embeddings-v5-omni-small`, an off-the-shelf open model you run as a sidecar, which does exactly that.

![A live Veda instance: one collection of sound clips and written notes. The query "a dog barking" is plain text, and the top result is the dog recording.](/videos/multimodal.mp4)

## A unified cross-modal space

`jina-embeddings-v5-omni-small` embeds **text, image, audio, and video** into the **same 1024-d cosine space**, so there's no per-modality model juggling and vectors across modalities are directly comparable. It's a third-party model — Veda doesn't train it; it registers it as a preset and routes each modality's request to it. Opt into the unified space by pointing **both** the text and the multimodal embedding config at the same omni sidecar:

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

The collection needs a vector field with `auto_embed { source: <blob field>, modality: image|audio|video }`, a `media_blob` blob field for the raw bytes, and optionally `source_path` / `class` string fields. A file whose modality has no matching `auto_embed` field returns `NO_MEDIA_FIELD` (`422`).

> **One media vector field per collection.** Each `auto_embed` vector field is pinned to **one** modality, and every dense vector field is a dense column that every record must fill — so a collection declaring both an image field and a video field can hold neither: whichever modality you upload leaves the other column empty. Give a media collection exactly one `auto_embed` vector field and use **one collection per modality**; `ingest-media` rejects a multi-vector-field collection up front with `INCOMPATIBLE_SCHEMA` (`409`). Cross-modal search still works across those collections because a unified model puts every modality in the same space.

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

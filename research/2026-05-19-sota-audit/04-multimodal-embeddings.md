# Veda SOTA Audit 04 — Multimodal Embeddings (2026-05-19)

## Verdict

**Qwen3-VL-Embedding-2B remains a defensible default in May 2026, but it is no longer the unambiguous SOTA it was at release (Jan 2026). It is currently a top-3 open-weights choice in the 1B-3B weight class and a top-5 choice overall.** The 2B variant scores **73.2 overall on MMEB-V2** (75.0 Image, 61.9 Video). The 8B sibling scores **77.8 overall** and was state-of-the-art on the MMEB-V2 leaderboard for several months. Both have since been edged out on the overall MMEB-V2 leaderboard by ByteDance's **Seed-1.6-Embedding** (API-only, 77.78 Image, 55.34 Video) and on the reasoning-heavy slices by research models like **TTEt-7B** (71.5 overall using a teacher-reasoner pipeline). The most important practical 2026 development is that the modality coverage frontier has moved from text+image to **text+image+document+video+audio in a single shared vector space**, where Veda's current model is only partial: Qwen3-VL covers text/image/document/video natively, but not audio.

**Top 3 alternatives:**

1. **Qwen3-VL-Embedding-8B** (Apache-2.0, same family) — drop-in upgrade path that lifts MMEB-V2 from 73.2 to 77.8, biggest gains on visual-document (82.4) and video (67.1). Adds about 4× the VRAM cost vs the 2B but stays inside one A100-40GB. The lowest-friction upgrade.
2. **Omni-Embed-Nemotron-3B** (NVIDIA OneWay Non-Commercial — *not Apache*; built on Qwen2.5-Omni-3B Thinker) — the cleanest open path to a **single shared embedding space for text + image + audio + video + document** that Veda has today. 2048-D embedding, 4.7B params, separate audio/video streams to avoid modality interference. License is the gating factor for any commercial use.
3. **Gemini Embedding 2** (Google, API-only, proprietary) — true 5-modality unified 3072-D space (text/image/video/audio/PDF) with Matryoshka down to 128-D. Strongest cross-lingual (0.997 Chinese↔English) and audio/video leadership. Best when you can tolerate a managed-API dependency.

**Recommendation:** *Keep Qwen3-VL-Embedding-2B as the default tier and add Qwen3-VL-Embedding-8B as an opt-in "quality tier".* The 2B → 8B upgrade is a one-config-flag move (same vLLM serving path, same tokenizer, same prompt template, same Apache-2.0 license, same Matryoshka behavior). Defer audio support — neither the model frontier nor the deployment story is settled enough in May 2026 to commit to a stack. When audio becomes a P0 requirement, evaluate Omni-Embed-Nemotron-3B (open weights, restrictive license) against Gemini Embedding 2 (proprietary, best quality) at that time. Do **not** switch to GME-Qwen2-VL-7B (it is a generation behind), and do **not** switch wholesale to Voyage or Cohere — they are credible but proprietary, and Veda's open-source positioning is part of its differentiation.

---

## 1. The MMEB-V2 Picture in May 2026

MMEB-V2 has become the canonical benchmark for multimodal retrieval and is the one that essentially every 2026 model is tuned and reported against. It spans **78 tasks across four input types** (text, image, video, visual document) and four meta-tasks (classification, VQA, retrieval, grounding). The leaderboard top in May 2026:

| Model | Params | Overall | Image | Video | VisDoc | License |
| --- | --- | --- | --- | --- | --- | --- |
| Seed-1.6-Embedding | n/a (API) | **~71.6** | **77.8** | **55.3** | 74.4 | Proprietary (Volcengine) |
| Qwen3-VL-Embedding-8B | 8B | 77.8 | 80.1 | 67.1 | **82.4** | Apache-2.0 |
| TTEt-7B (Think-Then-Embed) | 7B + 72B teacher | 71.5 | n/a | n/a | n/a | Research (Apache via Qwen2.5-VL) |
| VLM2Vec-V2 | 2B (Qwen2-VL) | 58.0 | n/a | n/a | n/a | Apache-2.0 |
| **Qwen3-VL-Embedding-2B** | **2B** | **73.2** | **75.0** | **61.9** | n/a | **Apache-2.0** |
| GME-Qwen2-VL-7B-Instruct | 8.3B | 67.4 (UMRB) | 65.8 (I→T) | n/a | 89.9 (T→VD) | Apache-2.0 |
| Voyage Multimodal 3.5 | n/a (API) | n/a (different bench) | 55.2 (video bench) | n/a | n/a | Proprietary |
| Cohere Embed v4 | n/a (API) | 65.2 (MTEB text) | n/a | n/a | n/a | Proprietary |
| Jina Embeddings v4 | 3.8B | 56.0 (MTEB-EN) | 90.2 (ViDoRe) | n/a | 90.2 (ViDoRe) | Open + API |
| Gemini Embedding 2 | n/a (API) | 68.2 (MTEB) | n/a | strong | n/a | Proprietary |

Two careful caveats: (a) Seed-1.6 leads on aggregate scores but it is **API-only** and **dual-modality (text/image/video)**, no audio; (b) the headline 77.8 from Qwen3-VL-8B was the leaderboard top as of Jan 8, 2025 per the Qwen paper — Seed-1.6 has been credited with edging past it on the Image slice since. The aggregate gap between Seed-1.6 and Qwen3-VL-8B is single-digit and reverses depending on which slice you weight.

**Key takeaway for Veda:** The 2B model Veda ships today is firmly in the second tier (~73). The 8B sibling is in the top-3. Switching to a non-Qwen model to chase Seed-1.6's marginal lead is not worth it because Seed is API-only and Veda's value proposition is open, self-hostable.

## 2. Family by Family

### 2.1 Qwen3-VL-Embedding (Alibaba, Jan 2026, what Veda runs)

Qwen3-VL-Embedding ships as **2B and 8B Apache-2.0 weights**. Both inherit Qwen3-VL's 30+ language coverage, 32k context, instruction conditioning, and dynamic-resolution vision tokenizer. Both expose **Matryoshka with output dims from 64 to 2048**, BF16, vLLM-native serving.

The 2B model is what Veda has wired up via vLLM today. Its measured strengths:

- **Smallest modality gap of any production model.** The cross-modal benchmark in Cheney Zhang's March 2026 study put Qwen3-VL-2B's modality gap at **0.25**, against Voyage's 0.59 and Gemini's 0.73. A small modality gap is what makes text↔image unified search actually work in a single vector index, which is Veda's design center. Qwen3-VL-2B's cross-modal hard_avg_R@1 of **0.945** topped that study, beating Gemini (0.928) and Voyage (0.900).
- **MMEB-V2 73.2 overall, 75.0 on Image, 61.9 on Video.** Strong for its size class.
- **No native audio.** This is the biggest 2026 gap for the family. Qwen3-VL-Omni (full multimodal LLM) exists but the *embedding* model derived from it does not, as of May 2026.
- **No native visual-document score reported** for the 2B variant in the paper — VisDoc results have been stronger for the 8B variant (82.4) than for the 2B.

The 8B model is the natural upgrade. Same API surface, same prompt format, same Apache-2.0 license, same vLLM path. The MMEB-V2 jump (73.2 → 77.8) is large for a one-tier scale-up, and the **VisDoc jump from "unreported" to 82.4** is the highest of any pure-dense model on that slice. VRAM cost is roughly 4× the 2B in BF16 (about 16-18 GB activations + KV at typical batch sizes), still single-GPU on an A100-40GB or H100. Quantization (INT8 / AWQ) drops it under 12 GB with <2% retrieval-accuracy loss.

### 2.2 GME (gme-Qwen2-VL-2B / 7B, Alibaba Tongyi Lab)

GME predates Qwen3-VL-Embedding by ~12 months. It is built on the older Qwen2-VL backbone, produces 3584-dim vectors at the 7B scale, and was the **previous SOTA on UMRB (67.44 overall)** before Qwen3-VL-Embedding superseded it on MMEB-V2. GME's surviving strengths are (a) very strong VisDoc (T→VD = 89.92 on UMRB) and (b) Apache-2.0 license. Its weaknesses are (c) single-image input only, capped at 1024 visual tokens, (d) English-only training (multilingual not guaranteed), and (e) being a generation behind on every modality.

**Verdict for Veda:** No reason to switch to GME from Qwen3-VL-Embedding. GME is Qwen's *previous* multimodal embedder; Veda is already on the successor.

### 2.3 Jina Embeddings v4 / Jina-CLIP v2

`jina-embeddings-v4` (3.8B, built on Qwen2.5-VL-3B-Instruct, April 2026) is the strongest open-weights model purpose-built for **visually rich document retrieval**. Headline numbers: **ViDoRe 90.17 nDCG@5**, Jina-VDR 84.11 single-vector / 80.2 multi-vector (yes, single-vector is higher on this slice), CLIP benchmark 84.1, MMTEB 66.49. It exposes both **dense single-vector (2048-D, Matryoshka down to 128)** and **late-interaction multi-vector (128-D per token)** modes from one model — a useful pattern Veda's `colbert` indexing track could exploit. 32k context, 29+ languages, supports images up to 20 megapixels. Free tier for evaluation.

`jina-clip-v2` (0.9B, Dec 2024) is much older but still relevant: 89-language multilingual, 512×512 image resolution, Matryoshka, and **strong CLIP-style throughput** for cases where you do not need a VLM-decoded encoder. CLIP-style models like jina-clip-v2 are 5-10× faster than VLM-based encoders for image-only encoding.

**Verdict for Veda:** Jina v4 is genuinely interesting *for the document-image use case* (ViDoRe 90.2 is best-in-class for dense single-vector). But it is built on Qwen2.5-VL while Veda is on Qwen3-VL — the natural upgrade direction is forward, not sideways. Worth considering if a *dedicated* document-image embedder is ever needed (e.g. a "documents" collection type), but not as the primary multimodal embedder.

### 2.4 Cohere Embed v4 (Multimodal)

Cohere's Embed v4 was the first **production-grade** model to put text, images, and *interleaved* mixed-modality documents (PDFs, slides, tables) into one space. **128k context** (longest of any multimodal embedder), Matryoshka at [256, 512, 1024, 1536]. Deployed via Cohere, AWS SageMaker, Azure AI Foundry. MTEB text-only score is 65.2 (behind Qwen3-Embedding-8B's 70.6 and Gemini Embedding 2's 68.3), but Cohere argues that's the wrong measurement because Embed v4 was the only production model that was natively multimodal at launch.

**Verdict for Veda:** Proprietary API; conflicts with Veda's self-hostable positioning. Useful as a comparison point but not a switch target.

### 2.5 Voyage Multimodal 3.5 (Voyage AI, Anthropic-affiliated, Jan 2026)

Voyage Multimodal 3.5 added **explicit video frame support** in Jan 2026, was the first production-grade video embedder with Matryoshka (2048, 1024, 512, 256). It claims **+30.57% over Google Multimodal Embedding 001, +2.26% over Cohere Embed v4, +8.38% over Amazon Nova 2 Multimodal** on a combined visual-document + video retrieval bench. On video specifically (Vatex, MSR-VTT, YouCook2) it scores 55.2, behind Amazon Nova 2 (60.3) and Gemini Embedding 2 (68.8). **No native audio.** Pricing is pixel + token based; first 200M tokens and 150B pixels free per account.

**Verdict for Veda:** Voyage is a credible second-place commercial multimodal embedder behind Gemini Embedding 2 on aggregate retrieval. But it is API-only and the modality gap is 2× worse than Qwen3-VL-2B's. Not worth the trade for Veda.

### 2.6 Gemini Embedding 2 (Google, March 2026)

The 2026 model to watch. **Five native modalities in one 3072-D Matryoshka space**: text (8k tokens), images (6 per request), videos (120s), audio (no transcription step), PDFs (6 pages). Truncatable to any dim 128–3072 without retraining. Matryoshka quality holds well: **68.16 @ 2048-D, 67.99 @ 768-D** on MTEB. Strongest cross-lingual scores in the field (0.997 Chinese↔English). Strongest reported video and audio retrieval. Customer-reported **70% latency reduction** by replacing Whisper-transcribe-then-embed audio pipelines with native audio embedding.

**Verdict for Veda:** Gemini Embedding 2 is *the* model that pushes the multimodal frontier in 2026, and it is what an organization should pick if API dependency is acceptable. For Veda's open-source value proposition, it is the wrong choice today. But it sets the bar that the open-weights frontier is now chasing — and Veda's "auto modality routing" architecture should anticipate audio being added as a fourth modality lane within 12 months.

### 2.7 NVIDIA: Omni-Embed-Nemotron-3B + Nemotron ColEmbed V2

NVIDIA released two distinct multimodal embedder families in 2025–2026:

- **Omni-Embed-Nemotron-3B** (Oct 2025, paper Sep 2025): a *dense* 2048-D embedder over **text + image + audio + video**. 4.7B params, built on Qwen2.5-Omni-3B's Thinker. ViDoRe v1 85.7 nDCG@5, LPM + FineVideo 0.706. **Separate audio and video streams** (not interleaved) to prevent modality interference. Critically: license is **NVIDIA OneWay Non-Commercial** + Qwen Research License — not Apache. Hardware: A100/H100 only.
- **Nemotron ColEmbed V2** (Feb 2026): **late-interaction (ColBERT-style)** document-image embedders at 3B/4B/8B, ranked #1/#3/#6 on ViDoRe V3 (8B at 63.42 nDCG@10). Per-token embeddings (4096-D for 8B). The current leader for visual-document late-interaction retrieval.

**Verdict for Veda:** Omni-Embed-Nemotron-3B is the most attractive *open-weights* path to native audio if and when Veda needs it — but its license is non-commercial, which conflicts with Veda's Apache-2.0 line. For document-image collections specifically, Nemotron ColEmbed V2 is the SOTA on ViDoRe V3 — but it is multi-vector and bigger to store. Veda's `colbert` index path would be a natural home for it as an opt-in "documents" backend.

### 2.8 Apple AIM / AIMv2

AIMv2 (Nov 2024, 3B) is a **vision encoder**, not an embedding model in the retrieval sense. It outperforms OpenAI CLIP and SigLIP on most vision-language understanding benchmarks (89.5% ImageNet-1k frozen-trunk for AIMv2-3B), but Apple has not shipped a contrastively-trained retrieval embedder built on it. Treat AIMv2 as a *future* vision-encoder backbone option for someone like Jina or NVIDIA to retrain, not as a candidate to switch to today.

### 2.9 CLIP-style baseline (OpenAI CLIP, EVA-CLIP, SigLIP, SigLIP-2)

CLIP-style dual-encoder models are now *baselines*, not frontier:

- **OpenAI CLIP ViT-L/14**: 768-D, the textbook baseline. Still the most-deployed image encoder.
- **EVA-CLIP-18B** (Feb 2024): 80.7% zero-shot ImageNet, the highest pure-CLIP score, but 18B params and English-only.
- **SigLIP-2** (Feb 2025): multilingual, four sizes (86M / 303M / 400M / 1B), and notably a 400M SigLIP-2 So400m **outperforms a 5.9B InternViT-6B as a VLM vision backbone**. SigLIP-2 is what most 2026 *open* multimodal LLMs (and several embedders) plug in as the vision encoder.

**Why CLIP-style is still relevant for Veda:** for image-only encoding where you do not need text-image fusion or instruction conditioning, CLIP/SigLIP-2 are 5-10× cheaper at inference than VLM-decoded embedders like Qwen3-VL-Embedding. Veda could expose a **"fast" image encoder tier** (SigLIP-2 So400m, 400M params, sub-millisecond on H100) for high-volume image-only collections, while keeping Qwen3-VL-Embedding for cross-modal queries. This is **not urgent** and should be ticket-on-demand, not pre-built.

### 2.10 ColPali / ColQwen2.5 (cross-reference to Agent 5)

Covered in detail in audit 05 (chunking / document pipelines). The relevant point for *embedding* selection: ColQwen2.5-7B was the previous ViDoRe leader; **Nemotron ColEmbed V2 8B** (Feb 2026) is the current leader. Late-interaction models are higher-recall on figure/table-heavy documents but **10–50× the storage cost** vs dense single-vector. Veda's current dense path (Qwen3-VL-Embedding) is the right *default*; the late-interaction path is the right *opt-in* for document-heavy workloads.

## 3. Axis-by-Axis Comparison

| Axis | Qwen3-VL-Emb-2B (Veda today) | Qwen3-VL-Emb-8B (upgrade) | Omni-Embed-Nemo-3B (audio) | Gemini Emb 2 (proprietary frontier) |
| --- | --- | --- | --- | --- |
| **MMEB-V2 overall** | 73.2 | 77.8 | n/a | n/a (68.2 MTEB) |
| **Image** | 75.0 | 80.1 | strong | strong |
| **Video** | 61.9 | 67.1 | 0.706 (LPM+FineVideo) | strongest |
| **VisDoc / ViDoRe** | n/a | 82.4 (MMEB-V2 VisDoc) | 85.7 (ViDoRe v1) | strong |
| **Audio** | none | none | yes (native) | yes (native) |
| **Dimensions** | 64–2048 (MRL) | 64–2048 (MRL) | 2048 fixed | 128–3072 (MRL) |
| **Context (text)** | 32k | 32k | 32k | 8k |
| **Vision encoder** | Qwen3-VL native VLM-decoded | Qwen3-VL native VLM-decoded | Qwen2.5-Omni Thinker | Gemini native |
| **Cross-lingual** | 0.988 (Zh↔En) | similar | n/a | 0.997 (best) |
| **Modality gap** | 0.25 (best) | similar | n/a | 0.73 (worst) |
| **GPU (BF16)** | ~6 GB single-A100 | ~18 GB single-A100 | A100-40G min | none (API) |
| **License** | Apache-2.0 | Apache-2.0 | NVIDIA OneWay NC | Proprietary API |
| **Serving** | vLLM native | vLLM native | TensorRT/Triton | Google API |
| **Unified space** | text+image+doc+video | text+image+doc+video | + audio | + audio, all five |

## 4. Modality-Specific SOTA in May 2026

- **Text + Image (general):** Qwen3-VL-Embedding-8B or Seed-1.6-Embedding (API). 8B has the smallest modality gap of any large model.
- **Document-image (page-level):** Nemotron ColEmbed V2 8B (late-interaction, ViDoRe V3 #1) or Jina Embeddings v4 (dense, ViDoRe 90.2). For dense single-vector at 3B scale, Jina v4 wins; for highest recall regardless of cost, Nemotron ColEmbed V2 8B wins.
- **Video retrieval:** Gemini Embedding 2 (68.8 on Vatex/MSR-VTT/YouCook2) > Amazon Nova 2 Multimodal (60.3) > Voyage Multimodal 3.5 (55.2). Open-weights: Qwen3-VL-Embedding-8B (67.1 MMEB-V2 Video) is competitive.
- **Audio retrieval (native, no Whisper transcription):** Gemini Embedding 2 (proprietary) or Omni-Embed-Nemotron-3B (open weights, non-commercial license). No Apache-2.0 native-audio embedder ships in May 2026.
- **Cross-lingual (multilingual retrieval):** Gemini Embedding 2 (0.997) ≈ Qwen3-VL-Embedding-2B (0.988) ≈ Jina v4 (0.985) ≈ Voyage MM-3.5 (0.982). All four are within 1.5%; pick on other axes.
- **Cross-modal (text ↔ image hard recall):** Qwen3-VL-Embedding-2B (0.945) > Gemini Embedding 2 (0.928) > Voyage MM-3.5 (0.900) > Jina CLIP v2 (0.873). Qwen leads here.

## 5. The Veda-Specific Story

Veda's "auto modality routing" — text-only → text model, anything else → multimodal model, both sharing one vector dim — is **exactly the right architecture for 2026**. The frontier is moving toward "one unified vector space across all modalities," and Veda is one routing-table edit away from supporting audio if and when it adds an audio-capable model. Concrete recommendations:

1. **Keep Qwen3-VL-Embedding-2B as the default multimodal model.** The cross-modal benchmark wins (0.945 hard_avg_R@1, 0.25 modality gap) directly justify Veda's "unified vector space" design.
2. **Add Qwen3-VL-Embedding-8B as an opt-in "quality" tier** in `config/embedding`. Same vLLM serving path, same prompts, same license. Document the VRAM cost (~3-4× the 2B). This gives users a one-config-line upgrade path from MMEB-V2 73.2 to 77.8 without changing the API.
3. **Track but don't chase Seed-1.6 / TTEt.** Seed-1.6 is API-only (Veda doesn't compete in the API-tier). TTEt-7B's 71.5 score uses a 72B teacher reasoner at *embedding time*, which is a 30× cost hit at query time and not suitable for production retrieval.
4. **Defer audio.** Neither the open-weights model (Omni-Embed-Nemotron-3B has a non-commercial license) nor the proprietary model (Gemini Embedding 2) is a clean drop-in. Wait for either Alibaba to ship a Qwen3-VL-Omni-Embedding (likely in H2 2026), or for NVIDIA to relicense Omni-Embed-Nemotron under Apache. Either is more likely than Veda picking a non-Apache model.
5. **For visual-document collections specifically, evaluate Jina Embeddings v4 as a dedicated backend.** ViDoRe 90.2 vs Qwen3-VL-8B's 82.4 (MMEB-V2 VisDoc) is a meaningful gap on PDF/slide retrieval. Two paths: dense single-vector (Jina v4 single-vec, slots into existing HNSW) or late-interaction (Jina v4 multi-vec, requires the `colbert` index path).
6. **Document the modality gap measurement** in Veda's research/ folder. The fact that Qwen3-VL-Embedding-2B has the smallest published modality gap (0.25) is a *real* competitive advantage vs Milvus/Qdrant/Weaviate defaults — most of those default to OpenAI text-embedding-3 + CLIP separately (different spaces, worst-case modality gap). Veda's single-space design should be marketed.
7. **Do not switch to GME** under any circumstance. It is Qwen2-VL based, English-only, single-image, and a strict generational step backward from what Veda runs today.

## 6. Sources

- [Qwen3-VL-Embedding and Qwen3-VL-Reranker (arXiv 2601.04720)](https://arxiv.org/abs/2601.04720)
- [Qwen3-VL-Embedding-2B Hugging Face card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Qwen3-VL-Embedding-8B Hugging Face card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- [GME-Qwen2-VL-7B-Instruct Hugging Face card](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct)
- [GME paper (arXiv 2412.16855)](https://arxiv.org/html/2412.16855v2)
- [Jina Embeddings v4 (Jina blog)](https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval/)
- [jina-embeddings-v4 paper (arXiv 2506.18902)](https://arxiv.org/abs/2506.18902)
- [Jina-CLIP v2 (Jina blog)](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/)
- [Cohere Embed Multimodal v4 changelog](https://docs.cohere.com/changelog/embed-multimodal-v4)
- [voyage-multimodal-3.5 announcement](https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/)
- [Gemini Embedding 2 announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Seed-1.6-Embedding (ByteDance)](https://seed1-6-embedding.github.io/)
- [Omni-Embed-Nemotron-3B Hugging Face card](https://huggingface.co/nvidia/omni-embed-nemotron-3b)
- [Omni-Embed-Nemotron paper (arXiv 2510.03458)](https://arxiv.org/html/2510.03458v1)
- [Nemotron ColEmbed V2 (NVIDIA blog)](https://huggingface.co/blog/nvidia/nemotron-colembed-v2)
- [VLM2Vec-V2 (arXiv 2507.04590)](https://arxiv.org/abs/2507.04590)
- [Think Then Embed (TTEt) (arXiv 2510.05014)](https://arxiv.org/html/2510.05014v2)
- [MIEB: Massive Image Embedding Benchmark (arXiv 2504.10471)](https://arxiv.org/abs/2504.10471)
- [SigLIP-2 (HuggingFace blog)](https://huggingface.co/blog/siglip2)
- [Apple AIMv2 paper (arXiv 2411.14402)](https://arxiv.org/abs/2411.14402)
- [EVA-CLIP-18B (arXiv 2402.04252)](https://arxiv.org/abs/2402.04252)
- [Embedding models benchmark 2026 — Cheney Zhang](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html)
- [Gemini Embedding 2 vs Qwen3 VL comparison (MindStudio)](https://www.mindstudio.ai/blog/gemini-embedding-2-vs-qwen3-vl-embeddings-comparison)
- [MMEB Leaderboard (HuggingFace Space)](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)

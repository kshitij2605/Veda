# Contributing to Veda

Thank you for your interest in contributing to Veda. This guide covers the development setup, coding standards, and pull request process.

## Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| Rust | 1.75+ | Core engine |
| Node.js | 22+ | Niriksha dashboard |
| Python | 3.9+ | Python SDK and tests |
| Docker | 24+ | Container builds and integration tests |

## Building

### Rust engine

```bash
cargo build --workspace
```

Release build:

```bash
cargo build --release
```

### Niriksha dashboard

```bash
cd niriksha
npm install
npm run build
```

For development with hot reload:

```bash
cd niriksha
npm run dev
```

### Python SDK

```bash
cd sdk/python
pip install -e ".[dev]"
```

## Testing

### Unit and integration tests

```bash
cargo test --workspace
```

### ANN benchmarks (longer running)

```bash
cargo test --test ann_benchmark -- --ignored --nocapture
```

### Niriksha tests

```bash
cd niriksha
npm run test
```

### Python SDK tests

```bash
cd sdk/python
pytest
```

## Code Style

### Rust

```bash
cargo fmt --all          # Format code
cargo clippy --workspace # Lint
```

All code must pass `cargo fmt --check` and `cargo clippy -- -D warnings` in CI.

### TypeScript (Niriksha)

```bash
cd niriksha
npm run lint
```

### Python

```bash
cd sdk/python
ruff check .
ruff format .
```

## Git Workflow

1. **Branch from `main`:**

   ```bash
   git checkout -b feat/<issue-number>-<short-description>
   # or
   git checkout -b fix/<issue-number>-<short-description>
   ```

2. **Use conventional commits:**

   ```
   feat: add jaccard distance metric
   fix: correct HNSW neighbor selection at layer 0
   docs: update API examples in README
   refactor: extract filter evaluator into separate module
   chore: update dependencies
   test: add stress test for concurrent inserts
   ```

3. **Commit after each meaningful change.** Do not batch unrelated changes into a single commit. Stage only the files relevant to the change.

4. **Never force push to `main`.** Never skip pre-commit hooks.

## Pull Request Process

Create PRs using the GitHub CLI:

```bash
gh pr create --title "feat: your change" --body "..."
```

Every PR description **must** include:

### Summary
1-3 bullet points describing what changed and why.

### Test Plan
A checklist of how the change was tested:
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed (describe)

### Requirements

- All CI checks must pass before merging.
- At least one approving review is required.
- Squash-merge to keep `main` history clean.
- Delete the feature branch after merging (both remote and local).

## Architecture Overview

| Crate | Description |
|---|---|
| `veda-core` | Shared types, distance functions (SIMD + scalar), filter expression parser and evaluator, chunking strategies, error types |
| `veda-index` | ANN indexes (HNSW, DiskANN, IVF-PQ, Flat), scalar indexes (inverted, range, bitmap), quantization (RaBitQ, SQ8) |
| `veda-storage` | mmap-backed vector storage, blob storage (MinIO/local), WAL abstraction, segment management |
| `veda-query` | Query planning, multi-vector fusion (RRF, WeightedSum, Intersection), distributed shard fan-out and merge |
| `veda-gateway` | REST API handlers (axum), JWT authentication, RBAC middleware, Prometheus metrics endpoint |
| `veda-coordinator` | Shard assignment, cluster topology, segment compaction, channel-based coordination |
| `veda-server` | Binary entry point, configuration loading, service startup and shutdown |

## Adding a New Chunking Strategy

1. **Add a variant** to `ChunkingStrategy` in `crates/veda-core/src/chunking.rs`.

2. **Implement the chunking logic** in the `chunk()` method's match arm. Your strategy receives the full document text and configuration (chunk_size, chunk_overlap) and returns a `Vec<String>` of chunks.

3. **Add unit tests** in the same module. At minimum, test:
   - Empty input returns an empty or single-element result
   - Output chunks are within the configured size limits
   - Overlap between consecutive chunks matches the configured overlap (if applicable)

4. **Register the strategy name** in the `FromStr` implementation so it can be selected via the `chunking_strategy` form parameter in the document processing endpoints.

## Adding a New Index Type

1. **Implement the `VectorIndex` trait** in `crates/veda-index/src/`. The trait requires:
   - `build()` -- construct the index from a set of vectors
   - `search()` -- return top-k nearest neighbors for a query vector
   - `insert()` -- add a vector to the index
   - `delete()` -- remove a vector by ID
   - `save()` / `load()` -- serialize and deserialize the index

2. **Add unit tests** in the same module. At minimum, test:
   - Build + search returns correct nearest neighbors on a small dataset
   - Insert after build is reflected in subsequent searches
   - Delete removes the vector from results
   - Save/load round-trips correctly

3. **Add a benchmark case** in the ANN benchmark suite.

4. **Register the index type** in the index factory so it can be selected via the API.

## Adding a New REST Endpoint

1. **Write the handler** in `crates/veda-gateway/src/handlers/`. Follow the existing pattern: extract request, validate, call into the query or storage layer, return a JSON response.

2. **Add the route** in the router configuration (`crates/veda-gateway/src/router.rs` or equivalent).

3. **Add integration tests** that exercise the endpoint through HTTP.

4. **Update the API documentation** if applicable.

## Questions?

Open an issue on GitHub or start a discussion. We are happy to help.

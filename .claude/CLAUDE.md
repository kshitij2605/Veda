# Project: Veda - Multimodal Vector Database

## Overview
Veda is a next-generation vector database designed for the multimodal era of RAG (Retrieval-Augmented Generation). It supports storing, indexing, and searching embeddings from any modality: text, images, audio, video, and documents. Includes **Niriksha**, a built-in data visualization and exploration dashboard (the "Kibana for multimodal vectors").

## Project Structure
```
project-mdb/
  .claude/          # Claude Code configuration
  research/         # Research documents and analysis
  docs/             # Design documents and plans
  crates/           # Rust workspace crates
    veda-core/      # Shared types, schema, vectors, SIMD, metrics, embedding
    veda-storage/   # Segments, vector storage, scalar indexes, WAL, blob storage
    veda-index/     # HNSW, DiskANN, IVF-PQ, Flat, RaBitQ, suffix array
    veda-query/     # Filter parser, search executor, fusion, distributed query
    veda-coordinator/ # Cluster state machine, shard router
    veda-gateway/   # REST API (axum), auth, RBAC
    veda-server/    # Binary entry point
  niriksha/         # Niriksha visualization dashboard (React/TypeScript)
  sdk/python/       # Python SDK (veda-client)
```

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## Architecture Principles

- Modular design: embedding, indexing, storage, query, and API layers should be decoupled
- Performance-first: minimize allocations and copies in hot paths
- Multimodal-native: not an afterthought bolted onto a text-only system
- Extensible: new modalities and index types should be easy to add

## Testing

- Unit tests for all core algorithms (distance functions, index operations)
- Integration tests for end-to-end query pipelines
- Benchmark tests for performance regression detection

## Git & GitHub Workflow

- **Branch naming:** `feat/<issue-number>-<short-description>` or `fix/<issue-number>-<short-description>`
- **Commit messages:** Conventional commits — `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`. Never include any mention of Claude Code, AI assistance, or "Co-Authored-By" lines in commit messages.
- **Commit after every major change:** Make a git commit after completing each meaningful unit of work. Don't batch unrelated changes into a single commit. Stage only the relevant files — not `git add .`.
- **Workflow:** Create GitHub issue -> feature branch -> implement with TDD -> PR -> CI passes -> merge -> delete branch
- **PRs:** Use `gh` CLI. PR description MUST include Summary and Test Plan sections.
- **CI:** All PRs must pass CI tests before merging. Never merge with failing checks.
- **Post-merge cleanup:** Delete feature branches both on remote and locally.
- **Never:** Force push to main, skip pre-commit hooks, or commit secrets/`.env` files.

## Tools & Automation

### Plugins (installed)
- **superpowers** — Workflow skills: brainstorming, writing-plans, TDD, systematic-debugging, code-review, git-worktrees, verification-before-completion. MUST use relevant superpowers skills before implementation work.
- **context7** — Up-to-date library documentation lookup. Use for checking API docs and code examples.
- **claude-md-management** — CLAUDE.md auditing and improvement.
- **playwright** — Browser automation for testing web UIs.

### MCP Servers
- **Context7** — Library documentation queries. Use `resolve-library-id` then `query-docs`.
- **IDE diagnostics** — Code diagnostics from the editor.

### Skills Usage
- **Before any non-trivial implementation:** Use `superpowers:brainstorming` -> `superpowers:writing-plans` -> then execute.
- **Before writing code:** Use `superpowers:test-driven-development` (RED -> GREEN -> REFACTOR).
- **For bugs:** Use `superpowers:systematic-debugging` before proposing fixes.
- **Before claiming done:** Use `superpowers:verification-before-completion` — evidence before assertions.
- **For code review:** Use `superpowers:requesting-code-review` after completing features.
- **For feature isolation:** Use `superpowers:using-git-worktrees` for work that needs isolation.

## Key Design Decisions

- Decisions are documented in `docs/plans/` as design documents
- Research findings are stored in `research/`

## Building & Running

- Build and run instructions will be added once the implementation language and build system are chosen

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation, and skills are invoked before action.

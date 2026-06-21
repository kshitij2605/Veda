# Contributing to Veda

Thank you for your interest in contributing to **Veda** and **Niriksha**. Contributions of
every size are welcome — a bug report, a doc fix, a new index type, or answering a question
all help make the project better.

## Ways to contribute

- **Report bugs** — file an issue with clear reproduction steps.
- **Request or add a feature** — a new index type, chunking strategy, modality, embedder,
  or REST endpoint. See the architecture notes in the repository `CONTRIBUTING.md`.
- **Improve the docs** — these pages live in `website/src/docs/content/` and are easy to edit.
- **Help the community** — answer questions in Discussions, review pull requests, help triage.
- **Spread the word** — write about Veda, share what you built, or ⭐ the repository.

New here? Look for issues labeled **`good first issue`** on GitHub.

## License & Contributor License Agreement

Veda is **dual-licensed**: the engine under **AGPL-3.0** and the Niriksha dashboard under
**FSL-1.1-ALv2** (see [Installation & config](/docs/installation) and the repository's
`LICENSING.md`). Because the project also offers a **commercial license**, every contribution
requires a signed **Contributor License Agreement (CLA)** so your code can be offered under
both the open-source and the commercial license.

- The **CLA bot prompts you automatically on your first pull request** — agreeing once covers
  future contributions.
- **You keep the copyright to your contributions** — the CLA grants the project a license, it
  does not transfer ownership.
- This differs from projects that use a DCO `Signed-off-by`; Veda specifically needs a CLA.

## Developing

```bash
git clone https://github.com/kshitij2605/Veda.git
cd Veda
```

### Build the engine (Rust)

The repo **pins its toolchain** via `rust-toolchain.toml` (currently **Rust 1.96.0**). With
`rustup` installed, `cargo` auto-selects the pinned version — you don't install a Rust version
by hand, and your `clippy`/`fmt` results match CI exactly.

```bash
cargo build --release                 # engine + server
cargo run --release -p veda-server    # run it (default ./veda-data 0.0.0.0:6340)
```

### Build the dashboard (Niriksha)

```bash
cd niriksha
npm install
npm run dev      # hot-reload dev server (proxies /v1 to the engine on :6340)
```

### Python SDK

```bash
cd sdk/python
pip install -e ".[dev]"
```

### These docs (the website)

```bash
cd website
npm install
npm run dev
```

Doc pages are plain Markdown in `website/src/docs/content/*.md`, registered in
`website/src/docs/registry.ts`. Add a file + a registry entry to add a page.

## Testing & linting

Run the same gates CI runs **before opening a PR** — they must pass:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings   # warnings are errors
cargo fmt --all --check
```

- **Dashboard:** `cd niriksha && npm run test && npm run lint`
- **Python SDK:** `cd sdk/python && pytest && ruff check . && ruff format --check .`
- **ANN benchmarks** (slower, optional): `cargo test --test ann_benchmark -- --ignored --nocapture`

Because the toolchain is pinned, local lint/format output is identical to CI — no "works on my
machine" lint drift.

## Issues

**Search existing issues first** to avoid duplicates. For a new bug report, include: reproduction
steps, expected vs. actual behavior, the Veda version/commit, and your environment.

**Security:** please do **not** open a public issue for a security vulnerability. Follow the
process in `SECURITY.md` (or email the maintainers privately) so it can be handled responsibly.

## Pull requests & code review

- **Branch from `main`:** `feat/<issue>-<short-desc>` or `fix/<issue>-<short-desc>`.
- **Conventional-commit PR titles:** `feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`,
  `chore:`, `ci:`. Keep commits focused; stage only the files relevant to the change.
- **Sign the CLA** (the bot prompts you on your first PR).
- **Every PR description must include a Summary and a Test Plan** (how you verified it).
- **Requirements to merge:** all CI checks green (clippy `-D warnings`, `fmt --check`, tests),
  at least one approving review, squash-merge to keep `main` clean, and delete the branch after.
- **Large or architectural changes** (new subsystems, on-disk format changes, big refactors):
  open an issue or discussion **first** to align on the design before a large PR.
- **AI-assisted contributions** are welcome **if** you are substantively involved, review and
  validate all generated code yourself, and disclose the assistance in the PR description.

## What to expect

This is a small team. We aim to respond to PRs and issues promptly, label what needs changes,
and keep you unblocked. If a PR stalls, a polite nudge on the thread is welcome.

## Thank you 🙏

Every contribution — code, docs, a bug report, or helping someone in Discussions — makes Veda
and Niriksha better for everyone. We're glad you're here.

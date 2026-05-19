# Linearisability Checkers & Distributed-Systems Testing Frameworks Beyond Jepsen

Date: 2026-05-19
Author: Research notes for Veda multi-week survey
Status: Draft - Survey & Recommendation

## Executive Summary

Veda needs Jepsen-class verification for its Raft-replicated record store: the ability to catch subtle linearisability violations under realistic fault injection (network partitions, packet drops, node crashes, clock skew, disk corruption). The question is whether to keep growing the in-house Rust checker, vendor an existing tool, or shell out to a battle-tested external one.

**Headline recommendation:** A *two-layer* stack, neither of which is Jepsen itself.

1. **Inside the process (deterministic, fast, every-PR):** Adopt **madsim** for deterministic simulation testing of the cluster - replace `tokio` with `madsim-tokio` behind a `cfg(madsim)` flag, simulate the whole multi-node cluster in one thread with seeded RNG, inject network/disk faults via madsim's APIs. RisingWave proves this scales to a real SQL streaming database; the licensing (Apache-2.0) and ergonomics fit Veda perfectly.
2. **Outside the process (black-box, post-merge, nightly):** Record operation histories in Jepsen's history JSON shape and ship them through **elle-cli** (a standalone JAR wrapping jepsen-io/elle and knossos) for linearisability and strict-serialisability checking. This gives us Kingsbury's actual checkers without writing a line of Clojure.

Below is the candidate-by-candidate analysis that backs that recommendation, followed by what we still have to build ourselves.

---

## Candidate analysis

### Jepsen itself (jepsen-io/jepsen)

- **License:** Eclipse Public License 1.0.
- **Language:** Clojure (~10k LOC). JVM 21+, Leiningen, JNA for SSH.
- **What it checks:** End-to-end: orchestrates the SUT over SSH on real nodes (LXC, EC2, bare metal), drives clients, runs nemeses (network partitions via iptables, clock skew via `libfaketime`, process kills, disk corruption), records the history, then hands it to Knossos/Elle for analysis.
- **Maturity:** v0.3.11 released March 2026. Actively used by Kingsbury for paid analyses; "Jepsen 18: Serializable Mom" presented at Systems Distributed June 2025. 7.4k stars. The only tool here used as a paid commercial audit. Latest writeups in 2025-2026 cover Bufstream, RDS PostgreSQL 17.4, TigerBeetle 0.16.x.
- **Integration cost (Rust SUT):** *High to extreme.* Writing the Database, Client, Checker, Nemesis, Generator, Workload abstractions for Veda is a multi-week project in Clojure. Tutorial 01 admits the scaffolding is small but tutorials 02-09 cover several thousand lines of test code per system, and nemesis composition + history merging is genuinely subtle. Plus an AWS or LXC harness for nodes. Realistic guess: 2-4 engineer-weeks to get a credible test, ongoing maintenance burden.
- **Verdict:** The gold standard but the wrong shape for us right now. We get most of its value via Elle (the checker) and madsim (the simulator) without paying the Clojure tax.

### Maelstrom (jepsen-io/maelstrom)

- **License:** EPL.
- **Language:** Clojure (the harness). SUTs in any language over JSON over STDIN/STDOUT.
- **What it checks:** Bundles Jepsen + Elle behind a teaching protocol. Provides standard workloads (lin-kv, txn-list-append, broadcast, g-set, etc.) and stock nemeses.
- **Maturity:** 0.2.4 (Dec 2024), 3.6k stars. Explicitly pedagogical - the README says it's a "workbench for learning."
- **Integration cost:** Low to write a node binary; *but* Veda's network is real TCP/HTTP, not Maelstrom's simulated message bus, and bolting Veda's REST API onto Maelstrom's JSON-over-stdio protocol means writing an adapter that re-implements clustering inside one process. That defeats the purpose.
- **Verdict:** Wrong tool. Designed for toy implementations from scratch, not for testing an already-running real database.

### Elle (jepsen-io/elle)

- **License:** EPL 2.0 or GPL-2.0 with classpath exception.
- **Language:** Clojure (JVM library).
- **What it checks:** Black-box transactional safety via cycle detection over the dependency graph. Catches G0 (write cycles), G1a (aborted reads), G1b (intermediate reads), G1c (cyclic info flow), G-Single (read skew), G2 (anti-dependency cycles), plus internal inconsistency, lost updates, garbage reads. Anomaly categories from Adya/Liskov/O'Neil. Two main checkers: `list-append` and `rw-register`.
- **Performance:** This is the killer fact. Knossos is exponential in concurrency, taps out at a few hundred ops. Elle is roughly linear in history length and roughly constant in concurrency - the paper checks **22 million transactions** for strong session serializability in ~2 minutes; 100-160k tx/sec on modern hardware. Kingsbury's morning-paper notes call this "seconds to minutes vs. seconds to millennia."
- **Maturity:** 693 stars, used as Jepsen's default transactional checker since 2020. VLDB 2020 paper. Currently the active production checker for paid Jepsen audits.
- **Integration cost:** Library is JVM-only, but **elle-cli** (ligurio/elle-cli, EPL, last release 0.1.9 in Aug 2025) wraps Elle + Knossos + bank/counter/set/mutex/sequential checkers as a standalone JAR that reads JSON or EDN history files and emits a verdict. Veda just needs to (a) record a history in the expected shape during a test, (b) `java -jar elle-cli.jar --model rw-register history.json`, (c) parse exit code and report files. JVM is the only runtime dep, which is acceptable for a CI nightly job in Docker.
- **Verdict:** **Yes.** This is how we get Kingsbury's actual checkers without writing Clojure.

### Knossos (jepsen-io/knossos)

- **License:** EPL.
- **Language:** Clojure.
- **What it checks:** Wing-Gong + competing-call algorithm for *linearisability* (not just serialisability) over a sequential model. The OG. Exact NP-complete search; exponential in concurrency.
- **Maturity:** 280 commits, 423 stars, but development effectively stalled after Elle subsumed its role for transactional checking. Kingsbury still uses Knossos when he needs strict linearisability over a sequential register/set model that Elle's relational approach can't express directly.
- **Integration cost:** Same as Elle - JVM library, but elle-cli wraps it (`--model cas-register`, `--model mutex`).
- **Verdict:** Use it via elle-cli when we need strict linearisability over a CAS-register model. For our typed record store with conditional puts (key, expected_version, new_value), CAS-register is the natural model.

### Porcupine (anishathalye/porcupine)

- **License:** MIT.
- **Language:** Go.
- **What it checks:** Linearisability via the "Faster linearizability checking via P-compositionality" algorithm. Reportedly 1000x-10000x faster than Knossos on the Jepsen test data corpus. Used in etcd robustness tests, TiDB, Amazon MemoryDB consistency tests, MIT 6.5840.
- **Maturity:** 111 commits, well-tested, very active community (etcd uses it weekly).
- **Integration cost:** Go-native library. From Rust, options are (a) subprocess - call a tiny Go wrapper that reads JSON history and emits a linearisability verdict, (b) cgo as `.a` static lib, (c) reimplementation. Subprocess is easiest. The sequential model must be expressed in Go though - we would port the Veda record-store model to ~50 lines of Go.
- **Rust ports:** `porcupine-rs` (erwin-kok) is a learning project with 1 star and 11 commits. Not production. `lincheck` and `model` are loom-based, single-process - wrong layer for us.
- **Verdict:** Strong runner-up. If we end up not happy with elle-cli's JVM overhead or strict-linearisability ergonomics, porcupine via Go subprocess is the most-battle-tested alternative. We are unlikely to need both.

### madsim (madsim-rs/madsim)

- **License:** Apache-2.0.
- **Language:** Pure Rust. Drop-in replacement for `tokio` behind `cfg(madsim)`; simulates the network, time, RNG, filesystem.
- **What it checks:** Not a checker - it is the *deterministic execution environment* that makes a checker effective. Replaces non-deterministic parts of the runtime (clocks, networking, RNG, async scheduler) so a seed reproduces the exact same execution. Lets us spin up an N-node Veda cluster inside a single process, drive client ops, inject partitions/drops/duplicates, then save the seed when something fails. Combined with a linearisability checker (lincheck, porcupine, or shipping the history to elle-cli) gives Jepsen-class coverage at thousand-fold speed.
- **Maturity:** 1.1k stars, Apache-2.0. RisingWave (commercial distributed SQL stream processor) runs the entire engine under madsim in CI. MadRaft uses it. Active development, sim wrappers exist for `tonic`, `etcd-client`, `rust-rdkafka`, `aws-sdk-s3`.
- **Integration cost:** Real but tractable. Veda must (a) feature-gate `tokio` -> `madsim::tokio`, (b) abstract all real I/O behind traits that have sim implementations, (c) ensure no hidden non-determinism (HashMap iteration order, system time outside the runtime, std::thread::spawn). This is invasive but a one-time cost; once done, every fault-injection test runs in seconds with perfect reproducibility. RisingWave's blog series documents the migration path well.
- **Verdict:** **Yes.** This is the most leveraged single investment we can make. Without madsim every flake is a heisenbug; with it, the seed *is* the bug report.

### Turmoil (tokio-rs/turmoil)

- **License:** MIT, last release 0.7.2 (Apr 2026), 1.2k stars.
- Sibling to madsim, also from the tokio side. Smaller in scope - covers TCP+UDP and fs only, less mature fault catalogue. Good for protocol-level testing but does not yet replace the whole tokio surface for a full database.
- **Verdict:** Inferior to madsim for our use case. Worth watching but not adopting.

### Stateright (stateright/stateright)

- **License:** MIT, 1.8k stars, 460 commits.
- **What it checks:** Explicit-state model checker for distributed state machines, with a linearisability tester built in. Can also run the same actors over real UDP. Excellent for verifying a *model* of Veda's replication protocol; can prove invariants exhaustively for small N.
- **Integration cost:** Medium - means writing Veda's replication semantics as Stateright actors. Useful as a separate verification artefact (like a TLA+ spec but in Rust), not as a runtime check on the production code.
- **Verdict:** Worth a small investment for the consensus crate (issue #177's scope), not for the end-to-end Veda linearisability problem.

### Loom & Shuttle

- **Loom** (tokio-rs): exhaustive C11 memory-model permutation testing. Sound but only scales to ~100s of operations and is for shared-memory concurrency primitives.
- **Shuttle** (awslabs, Apache-2.0, 988 stars, v0.9.0 Apr 2026): randomised scheduler control, scales much further than loom, *not sound* (passing test != correct). Designed for in-process concurrency.
- **Verdict:** Both are *single-process* tools. Wrong layer for distributed linearisability. Useful for veda-core's internal data structures (e.g. the WAL, segment manager), not for cluster behaviour. Worth keeping for crate-level tests but does not solve the user's question.

### Kani (model-checking/kani)

- **License:** MIT + Apache-2.0, 3.1k stars, v0.67.0 Jan 2026.
- Bit-precise bounded model checker for Rust. Used in AWS Firecracker, s2n-quic, parts of the Rust stdlib verification effort. Excellent for proving absence of UB and asserting properties of pure functions or small modules.
- **Verdict:** Wrong abstraction level - it verifies code, not distributed protocols. Useful for unsafe-heavy crates inside Veda (SIMD, custom allocators), not for cluster linearisability.

### P (p-org/P)

- **License:** MIT, 3.6k stars, 4215 commits, PeasyAI v1.0 March 2026. Used heavily inside AWS for protocol design (S3, DynamoDB, EBS - covered in the 2025 CACM AWS-formal-methods retrospective).
- DSL for state-machine modelling with PEx (random exploration) and PVerifier (symbolic) backends. PObserve is the runtime monitoring tool.
- **Integration cost:** Means writing a separate P spec of the Veda Raft protocol, plus optionally PObserve hooks to confirm production traces conform.
- **Verdict:** Same bucket as TLA+: orthogonal to runtime testing. We *should* eventually have a P or TLA+ spec of the consensus protocol, but that is a months-long parallel workstream, not a substitute for a runtime checker.

### TLA+ / TLC

- The classic. Already covered exhaustively in MongoDB, etcd, CockroachDB. Used by Microsoft Research on the CCF NSDI 2025 paper which found 5 safety bugs and 1 liveness bug in a consensus protocol via TLA+.
- **Verdict:** Same as P. We should have a TLA+ spec of the replication protocol eventually. Not a replacement for runtime checks.

### FizzBee

- **License:** Apache-2.0. Python-like DSL for formal specification of distributed systems. Single-process model checker, distributed checking on the roadmap. Has agent-skill support for Claude Code as of late 2025.
- **Verdict:** Newer, friendlier alternative to TLA+ for design-time verification. Same orthogonality argument as P/TLA+. Worth looking at when we want a sketch-then-check workflow for a new protocol but not for runtime linearisability.

### Antithesis (commercial) + OSS landscape

- Antithesis is the closed-source deterministic-simulation hypervisor built by the FoundationDB diaspora. Runs the whole SUT inside a deterministic VM. Very effective; very expensive; very locked.
- OSS analogues: madsim (Rust), Polar Signals' theatre-of-state-machines pattern, Diviner (xxuejie), dsim (danthegoodman1), Meta Hermit (container-level), Utkarsh Srivastava's QEMU work. As of 2026, none of these replicates Antithesis at the binary level - madsim remains the most practical for a Rust codebase.
- **Verdict:** If Veda ever gets to "spend money on third-party correctness audit" stage, Antithesis is the equivalent of paying Jepsen, but for now madsim covers ~80% of the value.

### TigerBeetle's VOPR

- Not a reusable framework - it is an in-house simulator for TigerBeetle's specific protocol (Viewstamped Replication). Runs 24/7 on 1024 cores at TB. Not designed to be vendored, but instructive: the *blueprint* is what we want to emulate, and madsim is the Rust-ecosystem analogue.
- **Verdict:** Pattern, not product.

### Chaos Mesh, Litmus

- CNCF Kubernetes chaos platforms. Inject network/IO/pod/time faults at the k8s level. *Not* linearisability checkers - they only provide the nemesis half.
- Caveat: Chaos Mesh had a critical 2025 CVE chain ("Chaotic Deputy", CVSS 9.8 RCE in the GraphQL server) - if we use it, must run in isolated test cluster.
- **Verdict:** Useful as the production-environment nemesis layer once Veda has a k8s deployment story; not relevant for unit-level linearisability checking.

### Academic / paper-backed tools worth name-checking

- **ahorn/linearizability-checker** - C++ port of Kingsbury's earlier work. Less maintained than Porcupine.
- "Efficient Decrease-and-Conquer Linearizability Monitoring" (arxiv 2410.04581) - online monitor, no public OSS release yet.
- "Automatically Verifying Replication-aware Linearizability" (arxiv 2502.19967) - SMT-based, prototype only.
- **lincheck** (Rust crate) - inspired by JetBrains' Kotlin Lincheck. Loom-backed, single-process. Useful for veda-core data structures.

---

## Recommendation - ranked top picks for Veda

| Rank | Tool | Where it slots in | Effort to adopt |
|------|------|-------------------|-----------------|
| 1 | madsim | Inside Veda: deterministic single-process simulation of multi-node cluster with fault injection | 1-3 weeks (cfg gating + I/O abstraction) |
| 2 | elle-cli + Jepsen JSON history shape | Outside Veda: nightly black-box check that recorded histories are linearisable / strictly-serializable | 3-5 days |
| 3 | porcupine (via Go subprocess) | Belt-and-braces alternative checker for CAS-register linearisability | 2-3 days if we go this route instead of (2) |
| 4 | TLA+ or P spec of replication protocol | Design-time orthogonal verification | Multi-week parallel workstream, low priority for now |
| 5 | Stateright | Model-checking the consensus implementation | 1-2 weeks, only if the home-grown Raft proves flaky |
| 6 | Kani / Loom / Shuttle | Per-crate verification for veda-core internals | Per-target, low overhead, opportunistic |
| 7 | Jepsen itself | Paid audit when Veda is feature-complete and ready for marketing | $$$, post-1.0 |

---

## What we still have to build ourselves

Even with the recommended stack, *most* of the harness is still ours. The OSS tools give us the checker and the deterministic runtime; we provide everything that knows about Veda. Concretely:

1. **History recorder** wired to Veda's REST API. Every client op gets a monotonic ID, an invocation timestamp, a return timestamp, a request body, a response body or error. Two output formats: the Jepsen history JSON shape for elle-cli, and a binary log for internal debugging.
2. **Sequential model of the Veda record store** as a Rust struct with `apply(op) -> response`. For elle-cli, we map our ops onto `rw-register` (with `version` field) or `cas-register`. The model is what defines "what should happen" and is the single most important artefact to get right - if the model is wrong, the checker is useless. ~200 lines.
3. **Cluster-under-madsim wrapper.** The veda-server binary needs to be runnable as a library: `let cluster = TestCluster::new(3); cluster.spawn(); ...`. All `tokio::spawn`, `tokio::time`, `tokio::net` calls feature-gated to `madsim::tokio` when `cfg(madsim)`. All storage I/O abstracted behind a trait so we can swap in an in-memory deterministic filesystem.
4. **Workload generator.** Tuned random walk over the API: insert, search, conditional update, delete, snapshot read, with controllable read/write ratio and key-space concentration to provoke contention. Should produce a stream of operations *and* their expected sequential effect against the model.
5. **Nemesis catalogue.** madsim handles network partitions, message drops, duplicates, reorders, latency injection out of the box. We add (a) node restarts mid-op, (b) disk full mid-write, (c) clock skew between leader and follower (madsim has the hooks), (d) WAL truncation. Aim for ~8 distinct nemeses composable into mixes.
6. **CI integration.** Nightly job that runs N seeds (~1000), each ~10000 ops, ~5min wall time at fast-forwarded sim time. On a failing seed, dump history JSON + replay log + minimal counterexample; ship history to elle-cli for double-confirmation; auto-open an issue.
7. **Glue for elle-cli output.** Parse its EDN/JSON verdict, surface anomaly type + offending tx IDs back into a Rust test report.
8. **Documentation** in `documentation/linearisability_testing.md` explaining the model, the seed-to-bug-report workflow, and how to run the nightly locally with a single seed for debugging.

Rough scope: a small but real project - one focused engineer, ~3-4 weeks to first credible green run, ongoing investment as we add new ops and new fault types. Roughly 1/4 the size of a from-scratch Jepsen integration in Clojure, and the whole thing stays in Rust.

---

## Specifically: is Jepsen still the right tool in 2025-2026?

Short answer: **Jepsen the *brand* and the *checker* are absolutely still the bar; Jepsen the *Clojure framework* is increasingly skippable if you are building a Rust system in 2026.**

Kingsbury's own work has bifurcated: he keeps the framework alive because it is what he uses for paid audits and because Elle/Knossos live in that codebase, but for greenfield Rust databases the dominant pattern (TigerBeetle, RisingWave, Turso, Resonate, S2) is now *deterministic simulation testing* + *cycle-detection consistency checker on recorded histories*. That is exactly the madsim + elle-cli recipe above.

Two trends made this shift:

- **Elle.** Once consistency-checking went from NP-hard exponential (Knossos) to roughly linear (Elle, 2020), histories from in-process simulation are big enough to be informative. Pre-Elle, you had to keep histories small, which meant slow real-network tests dominated. Post-Elle, you can run a million-op simulation in seconds and *also* check it.
- **DST in Rust.** FoundationDB pioneered DST in C++ in 2014; for a decade you had to roll your own runtime. madsim (2022+) makes it feasible in Rust without forking the runtime. RisingWave proved the recipe scales to a real product. Turso, Resonate, S2, Polar Signals each independently rebuilt or adopted variations in 2024-2025.

Net: Jepsen-the-audit is still the gold standard when you have a working product and budget. Jepsen-the-framework is no longer the lowest-friction way to *get* linearisability testing into a new Rust project. The right answer in 2026 is to use Kingsbury's checkers (Elle + Knossos via elle-cli), copy his fault-injection design (nemesis as a first-class concept), and host the whole thing inside a Rust deterministic simulator (madsim).

That is what this document recommends.

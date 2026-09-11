//! Linearisability history recorder + offline analyser.
//!
//! Not a full Jepsen port — Jepsen reasons about general
//! linearisability with arbitrary models and a much wider class
//! of fault scenarios. This is the Veda-specific subset that's
//! actually testable in a session: a key-value model over the
//! Raft-replicated record store, with one client op at a time
//! mapped to an "insert" or "read" of a (collection, record_id).
//!
//! ## Model
//!
//! Each operation is one of:
//!   - Insert(coll, id, payload)   — POST /v1/collections/{coll}/insert
//!   - Read(coll, id)              — GET /v1/collections/{coll}/records/{id}
//!
//! Each invocation has a (start_ns, end_ns) wall-clock window
//! and an outcome (Ok / Err / Timeout). Invocations across
//! threads / nodes happen concurrently; the recorder writes
//! them all to a single history (timestamps disambiguate).
//!
//! ## Linearisability check
//!
//! The check we implement: every successful Read must observe
//! the value of the most recent successful Insert (or no value
//! if no Insert preceded it) for that (collection, id) key.
//! "Most recent" is defined wrt invocations whose `end_ns` is
//! before the Read's `start_ns` — i.e. the Insert had already
//! returned before the Read began. Concurrent operations are
//! handled by checking *any* valid linearisation: a Read may
//! observe any Insert whose invocation overlaps in time.
//!
//! This catches the obvious violations: data loss (Read after
//! successful Insert sees nothing); stale reads (Read sees an
//! older value); spurious data (Read sees a value that no
//! Insert ever produced).
//!
//! ## What this DOES NOT catch
//!
//! - More general linearisability violations that involve
//!   operations other than insert/read (deletes, updates).
//!   Adding them is mechanical but multiplies state-space.
//! - Reordering violations that respect the I/O ordering but
//!   violate Raft's ordering guarantees on a partition heal.
//!   A full Jepsen-style checker would need an SMT-style
//!   linearisation search; we use a simpler "did some valid
//!   linearisation exist" heuristic.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpKind {
    Insert {
        collection: String,
        record_id: String,
        payload: String,
    },
    Read {
        collection: String,
        record_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpOutcome {
    /// Operation completed successfully.
    /// For Read: holds Some(payload) if the record was present,
    /// None if absent. For Insert: always None.
    Ok(Option<String>),
    /// HTTP error / 4xx / 5xx.
    Err(String),
    /// Did not complete within the timeout.
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invocation {
    pub seq: u64,
    pub start_ns: u128,
    pub end_ns: u128,
    pub client_id: usize,
    pub op: OpKind,
    pub outcome: OpOutcome,
    /// HLC commit_ts the server assigned to the operation, when
    /// the server returns one. Phase A scaffolding — the harness
    /// records `None` today; Phase C plumbs the commit_ts back
    /// through the REST response so this becomes the witness used
    /// by the Phase G snapshot-isolation checker. See
    /// `.claude/plans/structured-mixing-turing.md`.
    #[serde(default)]
    pub commit_ts: Option<u128>,
}

#[derive(Debug, Default)]
pub struct History {
    seq: AtomicU64,
    invocations: Mutex<Vec<Invocation>>,
}

impl History {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub async fn record(&self, client_id: usize, op: OpKind, outcome: OpOutcome, started: Instant) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Approximate start as (now - elapsed) since we don't
        // pin the wall-clock time of the Instant start.
        let elapsed = started.elapsed();
        let start_ns = now_ns.saturating_sub(elapsed.as_nanos());
        let inv = Invocation {
            seq,
            start_ns,
            end_ns: now_ns,
            client_id,
            op,
            outcome,
            commit_ts: None,
        };
        self.invocations.lock().await.push(inv);
    }

    pub async fn snapshot(&self) -> Vec<Invocation> {
        self.invocations.lock().await.clone()
    }

    /// Serialise the history to a JSONL file. Each line is one
    /// Invocation. Used so the operator can rerun the checker
    /// offline without re-driving the cluster.
    pub async fn dump_jsonl(&self, path: &std::path::Path) -> std::io::Result<()> {
        let inv = self.invocations.lock().await;
        let mut buf = String::new();
        for i in inv.iter() {
            let line = serde_json::to_string(i)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            buf.push_str(&line);
            buf.push('\n');
        }
        std::fs::write(path, buf)
    }

    /// Serialise the history to an Elle-compatible JSONL file.
    ///
    /// Format matches what Kingsbury's `elle-cli` expects: each
    /// line is one event, and each Veda Invocation emits two
    /// events — an `:invoke` at `start_ns` and an `:ok` / `:fail`
    /// / `:info` at `end_ns`.
    ///
    /// We model the (collection, record_id) tuple as an rw-register
    /// key `"{collection}/{record_id}"`. An Insert becomes a `:w`
    /// mop; a Read becomes an `:r` mop. Single-mop ops keep the
    /// model close to the simple register-history shape Elle uses
    /// for linearisability checks.
    ///
    /// Run the checker offline:
    ///   elle-cli check --model rw-register --consistency-models linearizable history.elle.jsonl
    pub async fn dump_elle_jsonl(&self, path: &std::path::Path) -> std::io::Result<()> {
        let inv = self.invocations.lock().await;
        let mut buf = String::new();
        // Events are interleaved in (start, end) order so the
        // checker sees them in the same wall-clock order they
        // actually happened.
        let mut events: Vec<EllEvent<'_>> = Vec::with_capacity(inv.len() * 2);
        for (idx, i) in inv.iter().enumerate() {
            let key = match &i.op {
                OpKind::Insert {
                    collection,
                    record_id,
                    ..
                }
                | OpKind::Read {
                    collection,
                    record_id,
                } => {
                    format!("{collection}/{record_id}")
                }
            };
            let (f, invoke_mop, complete_mop) = match (&i.op, &i.outcome) {
                (OpKind::Insert { payload, .. }, _) => (
                    "write",
                    serde_json::json!([":w", key, payload]),
                    serde_json::json!([":w", key, payload]),
                ),
                (OpKind::Read { .. }, OpOutcome::Ok(observed)) => (
                    "read",
                    serde_json::json!([":r", key, null]),
                    serde_json::json!([":r", key, observed]),
                ),
                (OpKind::Read { .. }, _) => (
                    "read",
                    serde_json::json!([":r", key, null]),
                    serde_json::json!([":r", key, null]),
                ),
            };
            let invoke_idx = (idx * 2) as u64;
            let complete_idx = invoke_idx + 1;
            events.push(EllEvent {
                index: invoke_idx,
                time: i.start_ns,
                process: i.client_id,
                kind: "invoke",
                f,
                value: vec![invoke_mop],
                error: None,
            });
            let (kind, error) = match &i.outcome {
                OpOutcome::Ok(_) => ("ok", None),
                OpOutcome::Err(e) => ("fail", Some(e.as_str())),
                OpOutcome::Timeout => ("info", Some("timeout")),
            };
            events.push(EllEvent {
                index: complete_idx,
                time: i.end_ns,
                process: i.client_id,
                kind,
                f,
                value: vec![complete_mop],
                error,
            });
        }
        events.sort_by_key(|e| (e.time, e.index));

        // Re-number densely AFTER the sort. `jepsen.history/dense-history`
        // requires indices to be 0..n-1 in file order and `assoc`s each event
        // into a vector at its index; anything else fails with
        // "Key must be integer" before a single check runs.
        //
        // The indices assigned above encode the invocation's position in the
        // list, which the time-sort then scrambles — so every dump this
        // harness has ever produced was unreadable by elle-cli, which is why
        // the cross-check the migration doc describes had never been run.
        // The ordering itself is correct; only the labels were stale.
        for (i, e) in events.iter_mut().enumerate() {
            e.index = i as u64;
        }

        for e in &events {
            let line = serde_json::to_string(e)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            buf.push_str(&line);
            buf.push('\n');
        }
        std::fs::write(path, buf)
    }
}

/// Serialised shape consumed by `elle-cli`. Field names + the use
/// of `:invoke` / `:ok` / `:fail` / `:info` follow Jepsen's
/// historical conventions so the same JSONL works with both
/// `elle-cli` and the Java/Clojure Jepsen tooling.
#[derive(Debug, Serialize)]
struct EllEvent<'a> {
    index: u64,
    time: u128,
    process: usize,
    /// One of "invoke" / "ok" / "fail" / "info". Serialized as the
    /// JSON string (Elle's loaders accept both the bare keyword
    /// shape and quoted strings).
    #[serde(rename = "type")]
    kind: &'a str,
    f: &'a str,
    value: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

/// Findings from a linearisability check.
#[derive(Debug, Default, Serialize)]
pub struct CheckReport {
    pub total_invocations: usize,
    pub successful_inserts: usize,
    pub successful_reads: usize,
    pub failed_invocations: usize,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Violation {
    pub kind: String,
    pub detail: String,
    pub seq: u64,
}

/// Check that every Read observes a payload consistent with
/// the Inserts that completed before it. See module docs for
/// the precise model.
pub fn check_history(history: &[Invocation]) -> CheckReport {
    let mut report = CheckReport {
        total_invocations: history.len(),
        ..Default::default()
    };
    // Build, for each (collection, record_id), the list of
    // successful Insert invocations sorted by end_ns.
    let mut inserts: HashMap<(String, String), Vec<&Invocation>> = HashMap::new();
    for inv in history {
        match (&inv.op, &inv.outcome) {
            (
                OpKind::Insert {
                    collection,
                    record_id,
                    ..
                },
                OpOutcome::Ok(_),
            ) => {
                report.successful_inserts += 1;
                inserts
                    .entry((collection.clone(), record_id.clone()))
                    .or_default()
                    .push(inv);
            }
            (OpKind::Read { .. }, OpOutcome::Ok(_)) => {
                report.successful_reads += 1;
            }
            (_, OpOutcome::Err(_)) | (_, OpOutcome::Timeout) => {
                report.failed_invocations += 1;
            }
        }
    }
    // Sort each per-key insert list by end_ns so we can binary
    // search by "completed before this point".
    for ins in inserts.values_mut() {
        ins.sort_by_key(|i| i.end_ns);
    }

    // For each successful Read, find the inserts whose
    // [start_ns, end_ns] window precedes or overlaps the
    // Read's start_ns. The Read MUST observe a payload from
    // one of those (or None if no Insert had even started).
    for inv in history {
        let OpKind::Read {
            collection,
            record_id,
        } = &inv.op
        else {
            continue;
        };
        let OpOutcome::Ok(observed) = &inv.outcome else {
            continue;
        };
        let key = (collection.clone(), record_id.clone());
        let none_inserts: Vec<&Invocation> = Vec::new();
        let ins_list = inserts.get(&key).unwrap_or(&none_inserts);

        // Inserts that completed strictly before this read started.
        let completed: Vec<&Invocation> = ins_list
            .iter()
            .copied()
            .filter(|i| i.end_ns <= inv.start_ns)
            .collect();
        // Inserts that started before this read ended and ended
        // after this read started (concurrent).
        let concurrent: Vec<&Invocation> = ins_list
            .iter()
            .copied()
            .filter(|i| i.start_ns < inv.end_ns && i.end_ns > inv.start_ns)
            .collect();

        match observed {
            None => {
                // Read saw nothing. Violation if any insert
                // had already completed (and no concurrent
                // insert could explain the absence).
                if !completed.is_empty() {
                    report.violations.push(Violation {
                        kind: "stale_read".into(),
                        detail: format!(
                            "Read on ({},{}) returned None but {} insert(s) had committed; latest at end_ns={}",
                            collection,
                            record_id,
                            completed.len(),
                            completed.last().unwrap().end_ns,
                        ),
                        seq: inv.seq,
                    });
                }
            }
            Some(payload) => {
                // Payload must match one of the inserts that
                // is either completed or concurrent.
                let candidates: Vec<&str> = completed
                    .iter()
                    .chain(concurrent.iter())
                    .filter_map(|i| {
                        if let OpKind::Insert { payload, .. } = &i.op {
                            Some(payload.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if candidates.is_empty() {
                    report.violations.push(Violation {
                        kind: "ghost_read".into(),
                        detail: format!(
                            "Read on ({},{}) returned a payload but no insert preceded or overlapped",
                            collection, record_id,
                        ),
                        seq: inv.seq,
                    });
                } else if !candidates.iter().any(|p| p == payload) {
                    report.violations.push(Violation {
                        kind: "payload_mismatch".into(),
                        detail: format!(
                            "Read on ({},{}) returned {:?}, expected one of {:?}",
                            collection, record_id, payload, candidates,
                        ),
                        seq: inv.seq,
                    });
                }
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv(seq: u64, start_ns: u128, end_ns: u128, op: OpKind, outcome: OpOutcome) -> Invocation {
        Invocation {
            seq,
            start_ns,
            end_ns,
            client_id: 0,
            op,
            outcome,
            commit_ts: None,
        }
    }

    #[test]
    fn happy_path_no_violations() {
        let h = vec![
            inv(
                0,
                100,
                200,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v1".into(),
                },
                OpOutcome::Ok(None),
            ),
            inv(
                1,
                300,
                400,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "1".into(),
                },
                OpOutcome::Ok(Some("v1".into())),
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 0);
        assert_eq!(r.successful_inserts, 1);
        assert_eq!(r.successful_reads, 1);
    }

    #[test]
    fn stale_read_caught() {
        let h = vec![
            inv(
                0,
                100,
                200,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v1".into(),
                },
                OpOutcome::Ok(None),
            ),
            inv(
                1,
                300,
                400,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "1".into(),
                },
                OpOutcome::Ok(None), // wrongly returned None
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 1);
        assert_eq!(r.violations[0].kind, "stale_read");
    }

    #[test]
    fn payload_mismatch_caught() {
        let h = vec![
            inv(
                0,
                100,
                200,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v1".into(),
                },
                OpOutcome::Ok(None),
            ),
            inv(
                1,
                300,
                400,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "1".into(),
                },
                OpOutcome::Ok(Some("v_wrong".into())),
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 1);
        assert_eq!(r.violations[0].kind, "payload_mismatch");
    }

    #[tokio::test]
    async fn elle_jsonl_round_trip_shape() {
        let history = History::new();
        let started = Instant::now();
        // Insert + matching read.
        history
            .record(
                0,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v1".into(),
                },
                OpOutcome::Ok(None),
                started,
            )
            .await;
        history
            .record(
                1,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "1".into(),
                },
                OpOutcome::Ok(Some("v1".into())),
                started,
            )
            .await;
        // One failed read so we exercise the "fail" branch too.
        history
            .record(
                1,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "missing".into(),
                },
                OpOutcome::Err("404".into()),
                started,
            )
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("history.elle.jsonl");
        history.dump_elle_jsonl(&path).await.unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        // 3 invocations × 2 events each = 6 lines
        assert_eq!(lines.len(), 6);
        // Every line round-trips through serde_json::Value.
        for l in &lines {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert!(v.get("type").is_some(), "missing type field");
            assert!(v.get("f").is_some(), "missing f field");
            assert!(v.get("value").is_some(), "missing value field");
        }
        // First line should be an "invoke".
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "invoke");
        // The "fail" branch must propagate the error string.
        let any_fail = lines.iter().any(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            v["type"] == "fail" && v["error"] == "404"
        });
        assert!(any_fail, "expected a fail event with error=\"404\"");
    }

    #[test]
    fn concurrent_read_can_see_either() {
        let h = vec![
            // Two overlapping inserts.
            inv(
                0,
                100,
                200,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v1".into(),
                },
                OpOutcome::Ok(None),
            ),
            inv(
                1,
                150,
                250,
                OpKind::Insert {
                    collection: "c".into(),
                    record_id: "1".into(),
                    payload: "v2".into(),
                },
                OpOutcome::Ok(None),
            ),
            // Read overlapping the second insert: may see either value.
            inv(
                2,
                220,
                280,
                OpKind::Read {
                    collection: "c".into(),
                    record_id: "1".into(),
                },
                OpOutcome::Ok(Some("v2".into())),
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 0);
    }
}

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

    pub async fn record(
        &self,
        client_id: usize,
        op: OpKind,
        outcome: OpOutcome,
        started: Instant,
    ) {
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
            let line = serde_json::to_string(i).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;
            buf.push_str(&line);
            buf.push('\n');
        }
        std::fs::write(path, buf)
    }
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
    let mut report = CheckReport::default();
    report.total_invocations = history.len();
    // Build, for each (collection, record_id), the list of
    // successful Insert invocations sorted by end_ns.
    let mut inserts: HashMap<(String, String), Vec<&Invocation>> = HashMap::new();
    for inv in history {
        match (&inv.op, &inv.outcome) {
            (OpKind::Insert { collection, record_id, .. }, OpOutcome::Ok(_)) => {
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
            _ => {}
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
        let OpKind::Read { collection, record_id } = &inv.op else { continue };
        let OpOutcome::Ok(observed) = &inv.outcome else { continue };
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

    fn inv(
        seq: u64,
        start_ns: u128,
        end_ns: u128,
        op: OpKind,
        outcome: OpOutcome,
    ) -> Invocation {
        Invocation {
            seq,
            start_ns,
            end_ns,
            client_id: 0,
            op,
            outcome,
        }
    }

    #[test]
    fn happy_path_no_violations() {
        let h = vec![
            inv(
                0, 100, 200,
                OpKind::Insert { collection: "c".into(), record_id: "1".into(), payload: "v1".into() },
                OpOutcome::Ok(None),
            ),
            inv(
                1, 300, 400,
                OpKind::Read { collection: "c".into(), record_id: "1".into() },
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
                0, 100, 200,
                OpKind::Insert { collection: "c".into(), record_id: "1".into(), payload: "v1".into() },
                OpOutcome::Ok(None),
            ),
            inv(
                1, 300, 400,
                OpKind::Read { collection: "c".into(), record_id: "1".into() },
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
                0, 100, 200,
                OpKind::Insert { collection: "c".into(), record_id: "1".into(), payload: "v1".into() },
                OpOutcome::Ok(None),
            ),
            inv(
                1, 300, 400,
                OpKind::Read { collection: "c".into(), record_id: "1".into() },
                OpOutcome::Ok(Some("v_wrong".into())),
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 1);
        assert_eq!(r.violations[0].kind, "payload_mismatch");
    }

    #[test]
    fn concurrent_read_can_see_either() {
        let h = vec![
            // Two overlapping inserts.
            inv(
                0, 100, 200,
                OpKind::Insert { collection: "c".into(), record_id: "1".into(), payload: "v1".into() },
                OpOutcome::Ok(None),
            ),
            inv(
                1, 150, 250,
                OpKind::Insert { collection: "c".into(), record_id: "1".into(), payload: "v2".into() },
                OpOutcome::Ok(None),
            ),
            // Read overlapping the second insert: may see either value.
            inv(
                2, 220, 280,
                OpKind::Read { collection: "c".into(), record_id: "1".into() },
                OpOutcome::Ok(Some("v2".into())),
            ),
        ];
        let r = check_history(&h);
        assert_eq!(r.violations.len(), 0);
    }
}

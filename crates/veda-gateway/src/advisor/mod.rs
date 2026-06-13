//! Veda-native ops advisor (AutoOps-style).
//!
//! This module is the *foundation* of the advisor: it only provides
//! the **snapshot collector** — a point-in-time picture of cluster
//! state that later rules and anomaly detectors evaluate. There are
//! no rules, events, REST routes, or notifications here yet; those
//! land in follow-up tasks.
//!
//! [`OpsSnapshot::collect`] reads live `AppState` (segment managers,
//! nodes, health probes, the recent query log, process RSS, and a
//! handful of Prometheus counters). [`SnapshotHistory`] is a bounded
//! ring buffer of snapshots that later detectors mine for rates and
//! trends.

pub mod snapshot;

pub use snapshot::{
    collect, parse_prom_counter, CollectionSnapshot, GlobalSnapshot, HealthSnapshot,
    NodeSnapshot, OpsSnapshot, QuerySnapshot, SnapshotHistory,
};

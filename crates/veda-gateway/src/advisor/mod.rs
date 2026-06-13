//! Veda-native ops advisor (AutoOps-style).
//!
//! The advisor turns a point-in-time [`snapshot::OpsSnapshot`] into
//! durable, actionable [`event::OpsEvent`]s:
//!
//! * [`snapshot`] — the **snapshot collector** + bounded history.
//! * [`rules`] — the built-in [`rules::Rule`] catalog; each rule maps a
//!   snapshot to zero or more [`rules::RuleHit`]s.
//! * [`event`] — the [`event::OpsEvent`] model + the persisted
//!   [`event::EventStore`] that reconciles hits into open/close
//!   transitions.
//! * [`config`] — per-rule [`config::RuleConfig`] knobs.
//!
//! Anomaly detection, notifications, auto-remediation, and the REST
//! surface land in follow-up tasks; nothing here sends, executes, or
//! exposes anything.

pub mod config;
pub mod event;
pub mod rules;
pub mod snapshot;

pub use snapshot::{
    collect, parse_prom_counter, CollectionSnapshot, GlobalSnapshot, HealthSnapshot,
    NodeSnapshot, OpsSnapshot, QuerySnapshot, SnapshotHistory,
};

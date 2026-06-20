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
//! * [`slo`] — the SLO engine: definitions, bucketed rolling counters,
//!   and SLI / error-budget / burn-rate math with JSON persistence.
//!
//! Anomaly detection, notifications, auto-remediation, and the REST
//! surface land in follow-up tasks; nothing here sends, executes, or
//! exposes anything.

pub mod anomaly;
pub mod changepoint;
pub mod config;
pub mod engine;
pub mod event;
pub mod faithfulness;
pub mod forecast;
pub mod incidents;
pub mod notify;
pub mod rca;
pub mod recall_probe;
pub mod remediation;
pub mod rules;
pub mod slo;
pub mod snapshot;
pub mod spike_attribution;
pub mod synthetic;

pub use engine::{AdvisorState, run_tick, spawn_advisor};

pub use slo::{
    BurnConfig, SloDef, SloKind, SloRegistry, SloState, SloStatus, catalog_default_slos,
    default_fast_burn, default_slow_burn,
};

pub use synthetic::{
    ProbeKind, ProbeResult, SyntheticProbe, SyntheticRegistry, SyntheticResults, SyntheticSummary,
    run_probe,
};

pub use recall_probe::{
    IndexParams, RecallResults, RecallSample, probe_recall, run_recall_probes, spawn_recall_prober,
};

pub use faithfulness::{FaithfulnessResults, FaithfulnessSample};

pub use forecast::{Forecast, TrendFit, compute_forecasts, linear_fit};

pub use changepoint::{ChangeKind, ChangePoint, compute_change_points, detect_change_point};

pub use spike_attribution::{Driver, SpikeAttribution, analyze_log_rate, attribute};

pub use incidents::{
    Incident, IncidentStore, cluster_incidents, compute_incidents, run_incident_correlation,
    spawn_incident_correlator,
};

pub use rca::{
    IncidentRef, LogLine, QueryLine, RcaBundle, ResourceState, correlate, correlate_incident,
};

pub use snapshot::{
    CollectionSnapshot, GlobalSnapshot, HealthSnapshot, NodeSnapshot, OpsSnapshot, QuerySnapshot,
    SnapshotHistory, collect, parse_prom_counter,
};

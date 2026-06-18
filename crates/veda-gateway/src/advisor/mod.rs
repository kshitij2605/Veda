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

pub use engine::{run_tick, spawn_advisor, AdvisorState};

pub use slo::{
    catalog_default_slos, default_fast_burn, default_slow_burn, BurnConfig, SloDef, SloKind,
    SloRegistry, SloState, SloStatus,
};

pub use synthetic::{
    run_probe, ProbeKind, ProbeResult, SyntheticProbe, SyntheticRegistry, SyntheticResults,
    SyntheticSummary,
};

pub use recall_probe::{
    probe_recall, run_recall_probes, spawn_recall_prober, IndexParams, RecallResults, RecallSample,
};

pub use faithfulness::{FaithfulnessResults, FaithfulnessSample};

pub use forecast::{compute_forecasts, linear_fit, Forecast, TrendFit};

pub use changepoint::{compute_change_points, detect_change_point, ChangeKind, ChangePoint};

pub use spike_attribution::{analyze_log_rate, attribute, Driver, SpikeAttribution};

pub use incidents::{
    cluster_incidents, compute_incidents, run_incident_correlation, spawn_incident_correlator,
    Incident, IncidentStore,
};

pub use rca::{correlate, correlate_incident, IncidentRef, LogLine, QueryLine, RcaBundle, ResourceState};

pub use snapshot::{
    collect, parse_prom_counter, CollectionSnapshot, GlobalSnapshot, HealthSnapshot,
    NodeSnapshot, OpsSnapshot, QuerySnapshot, SnapshotHistory,
};

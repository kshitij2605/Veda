//! Declarative per-field constraints, checked against records.
//!
//! This is SHACL's useful idea without SHACL, RDF, or a graph: a shape is
//! "field X must satisfy predicate P, and here is the violation report". The
//! ontology research (`research/2026-08-20-ontology-semantic-layer/`) found
//! this to be the one genuinely transferable piece of ontobricks' reasoning
//! layer — the rest of it needs a triple store this engine has no use for.
//!
//! # Scope, deliberately small
//!
//! Constraint validation appears **nowhere** in the fourteen ranked community
//! asks in `06-community-demand.md`; schema *evolution* was second and
//! validation was absent. So this ships the smallest set that covers real
//! validation and stops:
//!
//! - `required` — the value must be present and non-null
//! - `one_of` — the value must be one of an enumerated set
//! - `min` / `max` — numeric bounds, inclusive
//! - `max_length` — string length
//!
//! There is no regex, and that is a choice rather than an omission: it would
//! mean a new dependency in `veda-core` for a feature with no demand signal
//! behind it. If a pattern constraint is ever asked for, it is additive.
//!
//! Everything here is **pure**, so the rules are testable without a store.

use serde::{Deserialize, Serialize};

use crate::record::FieldValue;

/// Constraints on one field's values.
///
/// Empty by default, so a field that declares none is unconstrained and every
/// existing collection is unaffected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FieldConstraints {
    /// The field must be present and not [`FieldValue::Null`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub required: bool,
    /// Permitted values, compared as strings. Use for closed vocabularies —
    /// status codes, country codes, categories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<String>>,
    /// Inclusive numeric lower bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Inclusive numeric upper bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Maximum string length, in characters rather than bytes — a byte limit
    /// would reject a valid CJK value that a user counted as short.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

/// One failed constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Violation {
    pub field: String,
    /// Which rule failed: `required`, `one_of`, `min`, `max`, `max_length`.
    pub rule: String,
    /// Human-readable detail, safe to show an operator.
    pub detail: String,
}

impl FieldConstraints {
    /// Whether any constraint is set.
    pub fn is_empty(&self) -> bool {
        !self.required
            && self.one_of.is_none()
            && self.min.is_none()
            && self.max.is_none()
            && self.max_length.is_none()
    }

    /// Check `value` (absent = `None`) for `field`, collecting every failure.
    ///
    /// Returns **all** violations rather than the first, so one pass over a
    /// corpus tells an operator everything wrong with a record instead of
    /// making them fix and re-run repeatedly.
    pub fn check(&self, field: &str, value: Option<&FieldValue>) -> Vec<Violation> {
        let mut out = Vec::new();
        let v = |rule: &str, detail: String| Violation {
            field: field.to_owned(),
            rule: rule.to_owned(),
            detail,
        };

        let present = !matches!(value, None | Some(FieldValue::Null));
        if self.required && !present {
            out.push(v(
                "required",
                "field is required but missing or null".into(),
            ));
            // Every other rule is about a value, and there is none.
            return out;
        }
        if !present {
            // Absent and not required: unconstrained by the rest.
            return out;
        }
        let value = value.expect("present");

        if let Some(allowed) = &self.one_of {
            let s = as_comparable_string(value);
            match s {
                Some(s) if allowed.iter().any(|a| a == &s) => {}
                Some(s) => out.push(v(
                    "one_of",
                    format!("value {s:?} is not one of {allowed:?}"),
                )),
                // A vector or blob has no sensible string form; `one_of` on
                // such a field is a schema mistake, and saying so is more use
                // than silently passing.
                None => out.push(v(
                    "one_of",
                    "field type cannot be compared against an enumerated set".into(),
                )),
            }
        }

        if self.min.is_some() || self.max.is_some() {
            match as_number(value) {
                Some(n) => {
                    if let Some(min) = self.min
                        && n < min
                    {
                        out.push(v("min", format!("{n} is below the minimum {min}")));
                    }
                    if let Some(max) = self.max
                        && n > max
                    {
                        out.push(v("max", format!("{n} is above the maximum {max}")));
                    }
                }
                None => out.push(v(
                    "min",
                    "numeric bounds declared on a non-numeric field".into(),
                )),
            }
        }

        if let Some(limit) = self.max_length
            && let FieldValue::String(s) = value
        {
            // Characters, not bytes: a byte limit rejects valid CJK a user
            // counted as short.
            let n = s.chars().count();
            if n > limit {
                out.push(v(
                    "max_length",
                    format!("length {n} exceeds the maximum {limit}"),
                ));
            }
        }

        out
    }
}

/// String form for `one_of` comparison, or `None` when there isn't a sensible one.
fn as_comparable_string(v: &FieldValue) -> Option<String> {
    Some(match v {
        FieldValue::String(s) => s.clone(),
        FieldValue::Int32(n) => n.to_string(),
        FieldValue::Int64(n) => n.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Uuid(u) => u.to_string(),
        FieldValue::BlobRef(s) => s.clone(),
        // Floats are deliberately excluded: comparing an enumerated set
        // against a formatted float invites 0.1 != "0.1" surprises.
        _ => return None,
    })
}

fn as_number(v: &FieldValue) -> Option<f64> {
    Some(match v {
        FieldValue::Int32(n) => *n as f64,
        FieldValue::Int64(n) => *n as f64,
        FieldValue::Float32(n) => *n as f64,
        FieldValue::Float64(n) => *n,
        FieldValue::Timestamp(t) => *t as f64,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c() -> FieldConstraints {
        FieldConstraints::default()
    }

    #[test]
    fn an_unconstrained_field_never_violates() {
        // Every existing collection has this shape; validation must be a
        // total no-op there.
        assert!(c().is_empty());
        assert!(c().check("f", None).is_empty());
        assert!(
            c().check("f", Some(&FieldValue::String("anything".into())))
                .is_empty()
        );
    }

    #[test]
    fn required_catches_missing_and_null_but_not_empty_string() {
        let k = FieldConstraints {
            required: true,
            ..c()
        };
        assert_eq!(k.check("f", None)[0].rule, "required");
        assert_eq!(k.check("f", Some(&FieldValue::Null))[0].rule, "required");
        // An empty string is a value. Treating it as missing would silently
        // reject a legitimately blank field.
        assert!(
            k.check("f", Some(&FieldValue::String(String::new())))
                .is_empty()
        );
    }

    #[test]
    fn a_missing_optional_field_skips_the_other_rules() {
        // Otherwise an absent optional numeric would fail `min` for having no
        // value, which is not what the operator declared.
        let k = FieldConstraints {
            min: Some(10.0),
            one_of: Some(vec!["a".into()]),
            ..c()
        };
        assert!(k.check("f", None).is_empty());
    }

    #[test]
    fn one_of_accepts_members_and_reports_the_value_it_rejected() {
        let k = FieldConstraints {
            one_of: Some(vec!["EE".into(), "CL".into()]),
            ..c()
        };
        assert!(
            k.check("f", Some(&FieldValue::String("EE".into())))
                .is_empty()
        );
        let bad = k.check("f", Some(&FieldValue::String("Estonia".into())));
        assert_eq!(bad[0].rule, "one_of");
        assert!(
            bad[0].detail.contains("Estonia"),
            "the report must name the offending value: {}",
            bad[0].detail
        );
    }

    #[test]
    fn bounds_are_inclusive_and_report_both_ends() {
        let k = FieldConstraints {
            min: Some(0.0),
            max: Some(100.0),
            ..c()
        };
        assert!(k.check("f", Some(&FieldValue::Int64(0))).is_empty());
        assert!(k.check("f", Some(&FieldValue::Int64(100))).is_empty());
        assert_eq!(k.check("f", Some(&FieldValue::Int64(-1)))[0].rule, "min");
        assert_eq!(
            k.check("f", Some(&FieldValue::Float64(100.5)))[0].rule,
            "max"
        );
    }

    #[test]
    fn every_violation_is_reported_not_just_the_first() {
        // One pass over a corpus should tell an operator everything wrong with
        // a record, not make them fix and re-run.
        let k = FieldConstraints {
            one_of: Some(vec!["ok".into()]),
            max_length: Some(2),
            ..c()
        };
        let out = k.check("f", Some(&FieldValue::String("toolong".into())));
        let rules: Vec<&str> = out.iter().map(|v| v.rule.as_str()).collect();
        assert!(rules.contains(&"one_of"), "{rules:?}");
        assert!(rules.contains(&"max_length"), "{rules:?}");
    }

    #[test]
    fn max_length_counts_characters_not_bytes() {
        // A byte limit would reject a valid CJK value a user counted as short.
        let k = FieldConstraints {
            max_length: Some(3),
            ..c()
        };
        assert!(
            k.check("f", Some(&FieldValue::String("日本語".into())))
                .is_empty(),
            "three characters must pass a three-character limit"
        );
        assert_eq!(
            k.check("f", Some(&FieldValue::String("日本語だ".into())))[0].rule,
            "max_length"
        );
    }

    #[test]
    fn a_rule_on_an_incompatible_type_is_reported_rather_than_passed() {
        // Silently passing would let a schema mistake look like a clean corpus.
        let k = FieldConstraints {
            min: Some(1.0),
            ..c()
        };
        assert_eq!(
            k.check("f", Some(&FieldValue::String("x".into())))[0].rule,
            "min"
        );

        let e = FieldConstraints {
            one_of: Some(vec!["a".into()]),
            ..c()
        };
        assert_eq!(
            e.check(
                "f",
                Some(&FieldValue::Vector(crate::vector::DenseVector::new(vec![
                    0.0
                ])))
            )[0]
            .rule,
            "one_of"
        );
    }

    #[test]
    fn floats_are_not_compared_against_an_enumerated_set() {
        // Formatting a float to compare it invites 0.1 != "0.1" surprises.
        let k = FieldConstraints {
            one_of: Some(vec!["0.1".into()]),
            ..c()
        };
        assert_eq!(
            k.check("f", Some(&FieldValue::Float64(0.1)))[0].rule,
            "one_of",
            "a float must not silently match a string member"
        );
    }
}

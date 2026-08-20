//! Tune a collection's field annotations by measuring them.
//!
//! # Why this is not a dimension of the pipeline campaign
//!
//! The obvious design is to add `annotation_variant` to the tier-1 search
//! space beside `ef_search` and `top_k`. It does not work, for two reasons
//! that only show up when you look at what the campaign actually measures.
//!
//! 1. **The campaign's evaluator never runs an agent.** `evaluate_one_trial`
//!    issues a direct vector search per case, deliberately — an LLM call per
//!    (trial × case) would be prohibitive, and would conflate generation
//!    latency with the search latency the tuner optimises. Annotations act on
//!    how an *agent* builds its queries, so against a direct search they have
//!    **zero gradient** and NSGA-II would report noise as a result.
//! 2. **Annotations are not a per-trial choice.** A pipeline parameter is
//!    re-evaluated for every configuration; the right annotation for a schema
//!    does not depend on `ef_search`. Searching it inside the campaign would
//!    pay `variants × trials` agent runs to answer a `variants`-sized
//!    question.
//!
//! So this is a separate, one-shot pass: propose a handful of variants, score
//! each **once** against the agent eval set, keep the winner. Cost is
//! `(variants + 1)` eval runs, not `variants × trials`.
//!
//! # What it scores on
//!
//! The agent eval set (`rag_eval::run_eval`), which runs the real agent loop —
//! `describe_collection` included, which is the only path annotations reach.
//! The measured effect is large when it exists: on a schema whose stored
//! encodings differ from how questions phrase them (`ctry_cd` holding ISO-2
//! codes, `amt_2` holding cents), annotated scored 19/20 against 7/20
//! unannotated, and the wrong-encoding failure went from 11/20 to zero.
//!
//! It is **not** large on a self-describing schema, and this pass says so by
//! reporting the baseline beside every variant rather than only the winner. A
//! schema whose fields are already named `country` and `amount_usd` should be
//! expected to show no gain, and the right outcome then is to apply nothing.

use serde::{Deserialize, Serialize};
use veda_core::schema::{CollectionSchema, FieldAnnotations, Sensitivity};

/// One candidate set of annotations, keyed by field name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnnotationVariant {
    /// Human-readable label, e.g. `"baseline"` or `"with-value-domains"`.
    pub label: String,
    /// Field name -> annotations. Fields absent here keep what they have.
    pub fields: std::collections::BTreeMap<String, FieldAnnotations>,
}

impl AnnotationVariant {
    /// The do-nothing variant: every annotatable field stripped.
    ///
    /// Scored alongside the candidates so the report always answers "is this
    /// better than nothing?" rather than only "which candidate won?".
    pub fn baseline(schema: &CollectionSchema) -> Self {
        let fields = annotatable_fields(schema)
            .into_iter()
            .map(|name| (name, FieldAnnotations::default()))
            .collect();
        Self {
            label: "baseline (no annotations)".to_owned(),
            fields,
        }
    }
}

/// One variant's measured score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantScore {
    pub label: String,
    /// Mean answer relevancy over the eval set, the primary signal.
    pub answer_relevancy: Option<f64>,
    /// Fraction of cases meeting the eval set's pass bars.
    pub pass_rate: f64,
    /// Cases actually scored, so a thin result is visible rather than implied.
    pub case_count: usize,
}

/// Fields worth annotating: scalars an agent filters or reasons over.
///
/// Vector, sparse and multi-vector columns are excluded — an agent never
/// reasons about an embedding's contents, so describing one is cost without
/// signal, and it would dilute the prompt that the useful descriptions sit in.
pub fn annotatable_fields(schema: &CollectionSchema) -> Vec<String> {
    use veda_core::types::FieldType;
    schema
        .fields()
        .iter()
        .filter(|f| {
            !matches!(
                f.field_type(),
                FieldType::Vector { .. }
                    | FieldType::SparseVector { .. }
                    | FieldType::MultiVector { .. }
            )
        })
        .map(|f| f.name().to_owned())
        .collect()
}

/// Apply a variant to a schema, returning the modified copy.
///
/// Pure, so the rest of this module can be tested without a server. Fields the
/// variant does not mention are left exactly as they were: a variant that only
/// describes two columns must not silently strip a third that an operator
/// annotated by hand.
pub fn apply_variant(schema: &CollectionSchema, variant: &AnnotationVariant) -> CollectionSchema {
    let mut builder = CollectionSchema::builder(schema.name());
    for f in schema.fields() {
        let mut field = f.clone();
        if let Some(ann) = variant.fields.get(f.name()) {
            field = field.with_annotations(ann.clone());
        }
        builder = builder.add_field(field);
    }
    builder = builder
        .shard_count(schema.shard_count())
        .replication_factor(schema.replication_factor())
        .rls_owner_field(schema.rls_owner_field().map(str::to_owned));
    if let Some(fmt) = schema.stored_format() {
        builder = builder.stored_format(fmt);
    }
    if let Some(src) = schema.external() {
        builder = builder.external(src.clone());
    }
    // `build()` only fails on a schema that was already invalid, and this one
    // came from a live collection.
    builder.build().unwrap_or_else(|_| schema.clone())
}

/// Rank scores best-first, treating an unscored variant as worst.
///
/// Ties break toward the EARLIER entry, and the baseline is passed first, so a
/// variant that merely matches "no annotations" does not displace it. Applying
/// annotations that changed nothing is churn on a live schema.
pub fn rank(mut scores: Vec<VariantScore>) -> Vec<VariantScore> {
    scores.sort_by(|a, b| {
        let ka = (a.answer_relevancy.unwrap_or(f64::MIN), a.pass_rate);
        let kb = (b.answer_relevancy.unwrap_or(f64::MIN), b.pass_rate);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

/// Whether the winner is worth applying over the baseline.
///
/// Requires a margin, because the agent loop is nondeterministic: the
/// multi-hop work saw the same case answer correctly and incorrectly across
/// runs, and a campaign that applied every sub-noise "improvement" would
/// churn the schema for nothing. 0.02 on a [0,1] relevancy scale is small but
/// above the run-to-run spread observed on the annotation probe.
pub const MIN_IMPROVEMENT: f64 = 0.02;

pub fn worth_applying(winner: &VariantScore, baseline: &VariantScore) -> bool {
    match (winner.answer_relevancy, baseline.answer_relevancy) {
        (Some(w), Some(b)) => w - b >= MIN_IMPROVEMENT,
        // No relevancy signal at all: fall back to pass rate, same margin.
        _ => winner.pass_rate - baseline.pass_rate >= MIN_IMPROVEMENT,
    }
}

/// Draft annotations for `fields` from sampled values, via the answer-gen model.
///
/// Returns `None` when no model is configured — the pass then scores only the
/// variants the caller supplied, rather than failing.
pub async fn propose_variant(
    label: &str,
    fields: &[(String, String, Vec<String>)],
    with_value_domains: bool,
) -> Option<AnnotationVariant> {
    use veda_core::llm_client::{ChatMessage, CompletionOptions, LlmClient, LlmConfig};

    let client = LlmClient::new(LlmConfig::from_env_answer_gen()?);
    let described = fields
        .iter()
        .map(|(name, ty, samples)| {
            format!("- {name} ({ty}); example values: {}", samples.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let domain_rule = if with_value_domains {
        "Include a \"value_domain\" array of the distinct values when the field looks \
         categorical (fewer than about 20 distinct values); omit it otherwise."
    } else {
        "Do NOT include a \"value_domain\" key."
    };
    let prompt = format!(
        "These are columns of a search collection, with sample values:\n{described}\n\n\
         For each column return JSON: {{\"<field>\": {{\"description\": \"...\", \
         \"semantic_type\": \"...\"}}}}. description is one short sentence saying what the \
         column means. semantic_type is a short machine-readable tag for what the value IS \
         (for example iso3166_alpha2, email, currency_cents_usd, iso8601_date) — this matters \
         most when the stored encoding differs from how a person would phrase it. {domain_rule} \
         Return only the JSON object."
    );
    let out = client
        .complete(
            &[ChatMessage::text("user", prompt)],
            &CompletionOptions {
                temperature: Some(0.0),
                max_tokens: Some(1024),
                json_mode: true,
            },
        )
        .await
        .ok()?;
    parse_variant(label, &out.content)
}

/// Parse the model's JSON into a variant, skipping anything malformed.
///
/// Field-by-field rather than a whole-object derive: a model that gets one
/// column wrong should cost that column, not the whole variant.
pub fn parse_variant(label: &str, raw: &str) -> Option<AnnotationVariant> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let obj = v.as_object()?;
    let mut fields = std::collections::BTreeMap::new();
    for (name, spec) in obj {
        let Some(spec) = spec.as_object() else {
            continue;
        };
        let ann = FieldAnnotations {
            description: spec
                .get("description")
                .and_then(|d| d.as_str())
                .map(str::to_owned),
            semantic_type: spec
                .get("semantic_type")
                .and_then(|d| d.as_str())
                .map(str::to_owned),
            value_domain: spec.get("value_domain").and_then(|d| {
                let arr = d.as_array()?;
                Some(
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_owned))
                        .collect::<Vec<_>>(),
                )
            }),
            // Sensitivity is never inferred. Whether a column is restricted is
            // a policy decision an operator makes, and a model guessing it
            // would either leak a field it wrongly called public or hide one
            // it wrongly called restricted. Both are worse than not guessing.
            sensitivity: Sensitivity::Public,
            pii: false,
        };
        if ann.description.is_some() || ann.semantic_type.is_some() {
            fields.insert(name.clone(), ann);
        }
    }
    (!fields.is_empty()).then(|| AnnotationVariant {
        label: label.to_owned(),
        fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_core::schema::FieldDefinition;
    use veda_core::types::{DistanceMetric, FieldType};

    fn schema() -> CollectionSchema {
        CollectionSchema::builder("suppliers")
            .add_field(FieldDefinition::new("ctry_cd", FieldType::String).with_filterable(true))
            .add_field(FieldDefinition::new("amt_2", FieldType::Int64).with_filterable(true))
            .add_field(FieldDefinition::vector("vec", 8, DistanceMetric::Cosine))
            .build()
            .unwrap()
    }

    #[test]
    fn vector_columns_are_not_annotated() {
        // An agent never reasons about an embedding's contents; describing one
        // is cost without signal.
        assert_eq!(annotatable_fields(&schema()), vec!["ctry_cd", "amt_2"]);
    }

    #[test]
    fn applying_a_variant_leaves_unmentioned_fields_alone() {
        // A hand-annotated column must survive a variant that only describes
        // its neighbour.
        let mut s = CollectionSchema::builder("t")
            .add_field(
                FieldDefinition::new("kept", FieldType::String).with_annotations(
                    FieldAnnotations {
                        description: Some("set by hand".into()),
                        ..Default::default()
                    },
                ),
            )
            .add_field(FieldDefinition::new("changed", FieldType::String))
            .build()
            .unwrap();

        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "changed".to_owned(),
            FieldAnnotations {
                description: Some("from the model".into()),
                ..Default::default()
            },
        );
        s = apply_variant(
            &s,
            &AnnotationVariant {
                label: "v".into(),
                fields,
            },
        );

        assert_eq!(
            s.get_field("kept")
                .unwrap()
                .annotations()
                .description
                .as_deref(),
            Some("set by hand")
        );
        assert_eq!(
            s.get_field("changed")
                .unwrap()
                .annotations()
                .description
                .as_deref(),
            Some("from the model")
        );
    }

    #[test]
    fn the_baseline_strips_every_annotatable_field() {
        let s = apply_variant(&schema(), &AnnotationVariant::baseline(&schema()));
        for name in ["ctry_cd", "amt_2"] {
            assert!(s.get_field(name).unwrap().annotations().is_empty());
        }
    }

    fn score(label: &str, rel: Option<f64>, pass: f64) -> VariantScore {
        VariantScore {
            label: label.into(),
            answer_relevancy: rel,
            pass_rate: pass,
            case_count: 10,
        }
    }

    #[test]
    fn ranking_puts_the_best_relevancy_first_and_unscored_last() {
        let ranked = rank(vec![
            score("mid", Some(0.5), 0.5),
            score("unscored", None, 0.9),
            score("best", Some(0.9), 0.4),
        ]);
        assert_eq!(ranked[0].label, "best");
        assert_eq!(ranked[1].label, "mid");
        assert_eq!(
            ranked[2].label, "unscored",
            "a variant with no relevancy signal must not win on pass rate alone"
        );
    }

    #[test]
    fn a_sub_noise_improvement_is_not_worth_applying() {
        // The agent loop is nondeterministic; applying every tiny win churns a
        // live schema for nothing.
        let base = score("baseline", Some(0.80), 0.8);
        assert!(!worth_applying(&score("v", Some(0.81), 0.8), &base));
        assert!(worth_applying(&score("v", Some(0.83), 0.8), &base));
    }

    #[test]
    fn parsing_keeps_good_columns_and_drops_malformed_ones() {
        let v = parse_variant(
            "v",
            r#"{"ctry_cd": {"description": "Country.", "semantic_type": "iso3166_alpha2",
                            "value_domain": ["EE","CL"]},
                "amt_2": {"nonsense": 1},
                "broken": "not an object"}"#,
        )
        .unwrap();
        assert_eq!(v.fields.len(), 1, "only the well-formed column survives");
        let a = &v.fields["ctry_cd"];
        assert_eq!(a.semantic_type.as_deref(), Some("iso3166_alpha2"));
        assert_eq!(a.value_domain.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn sensitivity_is_never_inferred_by_the_model() {
        // A model guessing this either leaks a field it called public or hides
        // one it called restricted. Both are worse than not guessing.
        let v = parse_variant(
            "v",
            r#"{"ssn": {"description": "Social security number.",
                        "semantic_type": "ssn", "sensitivity": "restricted", "pii": true}}"#,
        )
        .unwrap();
        let a = &v.fields["ssn"];
        assert_eq!(a.sensitivity, Sensitivity::Public);
        assert!(!a.pii);
    }

    #[test]
    fn an_empty_or_junk_reply_yields_no_variant() {
        assert!(parse_variant("v", "not json").is_none());
        assert!(parse_variant("v", "{}").is_none());
        assert!(parse_variant("v", r#"{"f": {"unrelated": 1}}"#).is_none());
    }
}

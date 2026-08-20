//! End-to-end answer correctness, as a tuning objective.
//!
//! # Why this exists
//!
//! `TrialObjectives` scored `recall` and `faithfulness`, and **both are
//! retrieval metrics**. `recall` compares retrieved ids to expected ids;
//! `faithfulness`, despite the name, is
//! `cosine(embed(top_retrieved_chunk), embed(gold_answer))` — it never
//! generates an answer and never calls a generative model. So a change that
//! affects how a model *uses* what it retrieved — schema annotations, prompt
//! shape, anything downstream of the hit list — cannot move either objective.
//! Adding such a parameter to the search space would give NSGA-II a dimension
//! with **zero gradient**, and it would report noise as a result.
//!
//! This is not a Veda-specific observation. The RAG auto-tuning literature
//! reports the same split: "retrieval-optimal and RAG-optimal selection are
//! fundamentally distinct objectives"
//! ([Multi-Objective HPO for LLM and RAG Systems](https://arxiv.org/abs/2502.18635),
//! Barker et al.), which is why AutoRAG, RAGBuilder and Syftr all score the
//! generated answer rather than the retrieval alone.
//!
//! # What it measures
//!
//! RAGAS's `answer_correctness` decomposition, which combines an LLM's factual
//! judgement with embedding similarity to the reference answer:
//!
//! ```text
//! correctness = 0.75 * factual + 0.25 * similarity
//! ```
//!
//! - **factual** comes from the existing judge ([`super::rag_eval::judge_case`]):
//!   `claim_recall` if present, otherwise `1 - hallucination`. Both are already
//!   defined as "fraction of the answer's claims supported by the context".
//! - **similarity** is `cosine(embed(answer), embed(gold_answer))`, reusing the
//!   text embedder the eval harness already resolves.
//!
//! # Cost, and why it is opt-in
//!
//! Scoring an answer costs **two** model calls per case per trial — one to
//! generate, one to judge. A 50-trial campaign over a 20-case eval set is
//! 2,000 calls. It is therefore off by default and enabled per campaign; when
//! it is off the objective is `None` and is **excluded from the vector handed
//! to the optimizer**, rather than passed as a constant, so the search never
//! sees a flat dimension.

use veda_core::llm_client::{ChatMessage, CompletionOptions, LlmClient, LlmConfig};

/// RAGAS weighting. Factual agreement dominates: an answer that is fluent and
/// close to the reference in embedding space but wrong on the facts should not
/// score well.
const FACTUAL_WEIGHT: f64 = 0.75;
const SIMILARITY_WEIGHT: f64 = 0.25;

/// Generate an answer to `query` from `context`, using the answer-generation
/// model.
///
/// Returns `None` when no answer-generation endpoint is configured, which is
/// the same no-op posture eval-set generation takes — a campaign without the
/// model configured simply does not score answers.
pub async fn generate_answer(query: &str, context: &[String]) -> Option<String> {
    let client = LlmClient::new(LlmConfig::from_env_answer_gen()?);
    let joined = context.join("\n---\n");
    let messages = [
        ChatMessage::text(
            "system",
            "Answer the question using ONLY the provided context. Be concise and \
             factual. If the context does not contain the answer, say so plainly \
             rather than guessing.",
        ),
        ChatMessage::text("user", format!("Context:\n{joined}\n\nQuestion: {query}")),
    ];
    let opts = CompletionOptions {
        // Deterministic: a trial's score must not move because the generator
        // sampled differently, or the optimizer is chasing noise.
        temperature: Some(0.0),
        max_tokens: Some(256),
        json_mode: false,
    };
    client
        .complete(&messages, &opts)
        .await
        .ok()
        .map(|r| r.content)
}

/// Combine the judge's factual score and the embedding similarity.
///
/// Pure, so the weighting is testable without a model. `factual` is taken from
/// `claim_recall` when the judge supplied it, else `1 - hallucination`; a judge
/// that supplied neither yields `None` rather than a fabricated score.
pub fn combine(factual: Option<f64>, similarity: Option<f64>) -> Option<f64> {
    match (factual, similarity) {
        (Some(f), Some(s)) => Some(FACTUAL_WEIGHT * f + SIMILARITY_WEIGHT * s),
        // Either signal alone is still informative; rescale so the result
        // stays comparable on [0, 1] instead of silently capping at 0.75.
        (Some(f), None) => Some(f),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    }
}

/// Pull the factual score out of a judge verdict.
pub fn factual_from_verdict(v: &super::rag_eval::JudgeVerdict) -> Option<f64> {
    v.claim_recall.or_else(|| v.hallucination.map(|h| 1.0 - h))
}

/// Judge an answer against its context and reference, via the configured model.
async fn judge(
    answer: &str,
    context: &[String],
    gold: Option<&str>,
) -> Option<super::rag_eval::JudgeVerdict> {
    let client = LlmClient::new(LlmConfig::from_env_answer_gen()?);
    // `judge_case` takes an `FnOnce`, so the client moves straight in — no
    // clone, which `LlmClient` does not implement anyway.
    super::rag_eval::judge_case(answer, context, gold, move |prompt| {
        let prompt = prompt.to_owned();
        async move {
            let messages = [ChatMessage::text("user", prompt)];
            let opts = CompletionOptions {
                temperature: Some(0.0),
                max_tokens: Some(256),
                json_mode: true,
            };
            let out = client.complete(&messages, &opts).await.ok()?;
            parse_verdict(&out.content)
        }
    })
    .await
}

/// Parse the judge's JSON reply.
///
/// Done field by field rather than with a derive because `JudgeVerdict` is not
/// `Deserialize` and making it so is a change to the eval harness's public
/// shape for the sake of one caller. A missing or non-numeric field becomes
/// `None`, which [`combine`] already handles — a judge that returns partial
/// JSON degrades the score's coverage rather than failing the trial.
fn parse_verdict(raw: &str) -> Option<super::rag_eval::JudgeVerdict> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let get = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
    Some(super::rag_eval::JudgeVerdict {
        claim_recall: get("claim_recall"),
        context_utilization: get("context_utilization"),
        hallucination: get("hallucination"),
        self_knowledge: get("self_knowledge"),
    })
}

/// Generate an answer from `context` and judge it, returning both.
///
/// The **answer** is returned rather than a finished score so the caller can
/// compute the similarity half with the text embedder it has already resolved
/// — embedding it here would mean resolving a second embedder per case.
///
/// Returns `None` when no answer-generation model is configured or generation
/// failed. A campaign then carries no answer objective at all, rather than a
/// zero the optimizer would read as "this configuration is terrible".
pub async fn generate_and_judge(
    query: &str,
    context: &[String],
    gold: Option<&str>,
) -> Option<(String, Option<f64>)> {
    if context.is_empty() {
        return None;
    }
    let answer = generate_answer(query, context).await?;
    let factual = judge(&answer, context, gold)
        .await
        .as_ref()
        .and_then(factual_from_verdict);
    Some((answer, factual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::rag_eval::JudgeVerdict;

    #[test]
    fn combine_uses_the_ragas_weighting() {
        // 0.75 * 0.8 + 0.25 * 0.4 = 0.7
        let got = combine(Some(0.8), Some(0.4)).unwrap();
        assert!((got - 0.7).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn factual_dominates_a_fluent_but_wrong_answer() {
        // High embedding similarity, low factual agreement: the answer reads
        // like the reference and says something false. It must not outscore
        // the reverse case.
        let fluent_wrong = combine(Some(0.1), Some(0.9)).unwrap();
        let terse_right = combine(Some(0.9), Some(0.1)).unwrap();
        assert!(
            terse_right > fluent_wrong,
            "terse-but-correct {terse_right} should beat fluent-but-wrong {fluent_wrong}"
        );
    }

    #[test]
    fn one_signal_alone_is_not_silently_capped() {
        // Returning 0.75 * 1.0 for a perfect factual score with no gold answer
        // would make "no reference available" look like "somewhat wrong".
        assert_eq!(combine(Some(1.0), None), Some(1.0));
        assert_eq!(combine(None, Some(1.0)), Some(1.0));
        assert_eq!(combine(None, None), None);
    }

    #[test]
    fn a_partial_judge_reply_degrades_coverage_rather_than_failing() {
        let v = parse_verdict(r#"{"claim_recall": 0.8}"#).unwrap();
        assert_eq!(v.claim_recall, Some(0.8));
        assert_eq!(v.hallucination, None);
        assert_eq!(factual_from_verdict(&v), Some(0.8));

        assert!(parse_verdict("not json at all").is_none());
        // Present but non-numeric is treated as absent, not as zero.
        let bad = parse_verdict(r#"{"claim_recall": "high"}"#).unwrap();
        assert_eq!(bad.claim_recall, None);
    }

    #[test]
    fn factual_prefers_claim_recall_and_falls_back_to_hallucination() {
        let both = JudgeVerdict {
            claim_recall: Some(0.9),
            context_utilization: None,
            hallucination: Some(0.5),
            self_knowledge: None,
        };
        assert_eq!(factual_from_verdict(&both), Some(0.9));

        let only_hall = JudgeVerdict {
            claim_recall: None,
            context_utilization: None,
            hallucination: Some(0.25),
            self_knowledge: None,
        };
        assert_eq!(factual_from_verdict(&only_hall), Some(0.75));

        let neither = JudgeVerdict {
            claim_recall: None,
            context_utilization: None,
            hallucination: None,
            self_knowledge: None,
        };
        assert_eq!(
            factual_from_verdict(&neither),
            None,
            "a judge that scored nothing must not yield a fabricated number"
        );
    }
}

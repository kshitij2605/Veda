use std::collections::HashMap;

use roaring::RoaringBitmap;
use veda_core::error::{Result, VedaError};
use veda_storage::bitmap_index::BitmapIndex;
use veda_storage::inverted_index::InvertedIndex;
use veda_storage::range_index::RangeIndex;

use super::ast::{FilterExpr, FilterValue};

/// Context for evaluating filter expressions against scalar indexes.
///
/// Holds references to the inverted, range, and bitmap indexes for each field,
/// plus the total document count (needed for NOT / IS NULL operations).
pub struct FilterContext<'a> {
    inverted_indexes: HashMap<String, &'a InvertedIndex>,
    range_indexes: HashMap<String, &'a RangeIndex>,
    bitmap_indexes: HashMap<String, &'a BitmapIndex>,
    total_docs: u32,
}

impl<'a> FilterContext<'a> {
    /// Create a new evaluation context.
    ///
    /// `total_docs` is the total number of documents in the collection. It is
    /// used to construct the universe bitmap for NOT and IS NULL operations.
    pub fn new(total_docs: u32) -> Self {
        Self {
            inverted_indexes: HashMap::new(),
            range_indexes: HashMap::new(),
            bitmap_indexes: HashMap::new(),
            total_docs,
        }
    }

    /// Register an inverted index for the given field.
    pub fn add_inverted(&mut self, field: &str, index: &'a InvertedIndex) {
        self.inverted_indexes.insert(field.to_owned(), index);
    }

    /// Register a range index for the given field.
    pub fn add_range(&mut self, field: &str, index: &'a RangeIndex) {
        self.range_indexes.insert(field.to_owned(), index);
    }

    /// Register a bitmap index for the given field.
    pub fn add_bitmap(&mut self, field: &str, index: &'a BitmapIndex) {
        self.bitmap_indexes.insert(field.to_owned(), index);
    }

    /// Evaluate a filter expression, returning matching document IDs.
    pub fn evaluate(&self, expr: &FilterExpr) -> Result<RoaringBitmap> {
        match expr {
            FilterExpr::Eq { field, value } => self.eval_eq(field, value),
            FilterExpr::Ne { field, value } => {
                let eq_result = self.eval_eq(field, value)?;
                Ok(self.universe() - eq_result)
            }
            FilterExpr::Gt { field, value } => self.eval_comparison(field, value, CmpOp::Gt),
            FilterExpr::Gte { field, value } => self.eval_comparison(field, value, CmpOp::Gte),
            FilterExpr::Lt { field, value } => self.eval_comparison(field, value, CmpOp::Lt),
            FilterExpr::Lte { field, value } => self.eval_comparison(field, value, CmpOp::Lte),
            FilterExpr::Between { field, low, high } => self.eval_between(field, low, high),
            FilterExpr::In { field, values } => self.eval_in(field, values),
            FilterExpr::Contains { field, value } => self.eval_contains(field, value),
            FilterExpr::StartsWith { field, value } => self.eval_starts_with(field, value),
            FilterExpr::IsNull { field } => self.eval_is_null(field),
            FilterExpr::IsNotNull { field } => self.eval_is_not_null(field),
            FilterExpr::And(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                Ok(l & r)
            }
            FilterExpr::Or(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                Ok(l | r)
            }
            FilterExpr::Not(inner) => {
                let result = self.evaluate(inner)?;
                Ok(self.universe() ^ result)
            }
        }
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Build the universe bitmap: all doc IDs from 0..total_docs.
    fn universe(&self) -> RoaringBitmap {
        RoaringBitmap::from_sorted_iter(0..self.total_docs).unwrap()
    }

    /// Evaluate an equality comparison.
    fn eval_eq(&self, field: &str, value: &FilterValue) -> Result<RoaringBitmap> {
        // Try range index for numeric values.
        if let Some(f) = value.as_f64()
            && let Some(idx) = self.range_indexes.get(field)
        {
            return Ok(idx.search_eq(f));
        }

        let str_val = value.as_str_repr();

        // Try bitmap index.
        if let Some(idx) = self.bitmap_indexes.get(field) {
            return Ok(idx.search_eq(&str_val).cloned().unwrap_or_default());
        }

        // Try inverted index.
        if let Some(idx) = self.inverted_indexes.get(field) {
            return Ok(idx.search_exact(&str_val).cloned().unwrap_or_default());
        }

        Err(VedaError::Query(format!(
            "no index found for field '{field}'"
        )))
    }

    /// Evaluate a range comparison (Gt, Gte, Lt, Lte).
    fn eval_comparison(
        &self,
        field: &str,
        value: &FilterValue,
        op: CmpOp,
    ) -> Result<RoaringBitmap> {
        let f = value.as_f64().ok_or_else(|| {
            VedaError::Query(format!(
                "range comparison requires numeric value, got {value:?}"
            ))
        })?;

        let idx = self
            .range_indexes
            .get(field)
            .ok_or_else(|| VedaError::Query(format!("no range index for field '{field}'")))?;

        Ok(match op {
            CmpOp::Gt => idx.search_gt(f),
            CmpOp::Gte => idx.search_gte(f),
            CmpOp::Lt => idx.search_lt(f),
            CmpOp::Lte => idx.search_lte(f),
        })
    }

    /// Evaluate a BETWEEN expression.
    fn eval_between(
        &self,
        field: &str,
        low: &FilterValue,
        high: &FilterValue,
    ) -> Result<RoaringBitmap> {
        let lo = low.as_f64().ok_or_else(|| {
            VedaError::Query(format!("BETWEEN requires numeric values, got {low:?}"))
        })?;
        let hi = high.as_f64().ok_or_else(|| {
            VedaError::Query(format!("BETWEEN requires numeric values, got {high:?}"))
        })?;

        let idx = self
            .range_indexes
            .get(field)
            .ok_or_else(|| VedaError::Query(format!("no range index for field '{field}'")))?;

        Ok(idx.search_range(lo, hi))
    }

    /// Evaluate an IN expression.
    fn eval_in(&self, field: &str, values: &[FilterValue]) -> Result<RoaringBitmap> {
        let str_vals: Vec<String> = values.iter().map(|v| v.as_str_repr()).collect();
        let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();

        // Try bitmap index first.
        if let Some(idx) = self.bitmap_indexes.get(field) {
            return Ok(idx.search_in(&str_refs));
        }

        // Try inverted index.
        if let Some(idx) = self.inverted_indexes.get(field) {
            return Ok(idx.search_in(&str_refs));
        }

        Err(VedaError::Query(format!(
            "no index found for field '{field}'"
        )))
    }

    /// Evaluate a CONTAINS (substring) expression.
    fn eval_contains(&self, field: &str, substring: &str) -> Result<RoaringBitmap> {
        if let Some(idx) = self.inverted_indexes.get(field) {
            return Ok(idx.search_contains(substring));
        }

        Err(VedaError::Query(format!(
            "no inverted index for field '{field}' (CONTAINS requires inverted index)"
        )))
    }

    /// Evaluate a STARTS_WITH (prefix) expression.
    fn eval_starts_with(&self, field: &str, prefix: &str) -> Result<RoaringBitmap> {
        if let Some(idx) = self.inverted_indexes.get(field) {
            return Ok(idx.search_prefix(prefix));
        }

        Err(VedaError::Query(format!(
            "no inverted index for field '{field}' (STARTS_WITH requires inverted index)"
        )))
    }

    /// Evaluate IS NULL: docs that have no indexed value for the field.
    fn eval_is_null(&self, field: &str) -> Result<RoaringBitmap> {
        let indexed = self.all_indexed_docs(field)?;
        Ok(self.universe() - indexed)
    }

    /// Evaluate IS NOT NULL: docs that have at least one indexed value.
    fn eval_is_not_null(&self, field: &str) -> Result<RoaringBitmap> {
        self.all_indexed_docs(field)
    }

    /// Collect the union of all doc IDs indexed for a given field.
    fn all_indexed_docs(&self, field: &str) -> Result<RoaringBitmap> {
        if let Some(idx) = self.inverted_indexes.get(field) {
            return Ok(idx.all_doc_ids());
        }
        if let Some(idx) = self.bitmap_indexes.get(field) {
            return Ok(idx.search_in(&idx.distinct_values()));
        }
        if let Some(idx) = self.range_indexes.get(field) {
            // RangeIndex doesn't have an all_doc_ids helper, so use full range.
            return Ok(idx.search_range(f64::NEG_INFINITY, f64::INFINITY));
        }
        Err(VedaError::Query(format!(
            "no index found for field '{field}'"
        )))
    }
}

/// Internal enum for comparison dispatch.
enum CmpOp {
    Gt,
    Gte,
    Lt,
    Lte,
}

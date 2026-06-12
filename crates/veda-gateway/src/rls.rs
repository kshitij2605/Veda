//! Row-Level Security helpers for per-collection owner-field scoping.

use axum::http::StatusCode;
use axum::Json;
use veda_query::filter::ast::{FilterExpr, FilterValue};

use crate::auth::{Claims, Role};
use crate::rest::collections::ErrorResponse;

/// Produce the effective filter for a read, injecting an owner-equality
/// predicate for non-Admin callers on RLS-enabled collections.
///
/// If `is_admin` is true, returns `user_filter` unchanged.
/// Otherwise ANDs `Eq { owner_field == user }` onto `user_filter`.
pub fn inject_owner_filter(
    user_filter: Option<FilterExpr>,
    owner_field: &str,
    user: &str,
    is_admin: bool,
) -> Option<FilterExpr> {
    if is_admin {
        return user_filter;
    }
    let owner_eq = FilterExpr::Eq {
        field: owner_field.to_owned(),
        value: FilterValue::String(user.to_owned()),
    };
    match user_filter {
        None => Some(owner_eq),
        Some(existing) => Some(FilterExpr::And(Box::new(existing), Box::new(owner_eq))),
    }
}

/// Enforce the owner field on a JSON record map before insertion.
///
/// - Omitted owner field: auto-set to `claims.sub`.
/// - Owner field set to a different value than `claims.sub`: 403.
/// - Admin callers: no restrictions.
pub fn enforce_owner_on_insert(
    record: &mut serde_json::Map<String, serde_json::Value>,
    owner_field: &str,
    claims: &Claims,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if claims.role == Role::Admin {
        return Ok(());
    }
    let user = &claims.sub;
    match record.get(owner_field) {
        None => {
            record.insert(owner_field.to_owned(), serde_json::Value::String(user.clone()));
            Ok(())
        }
        Some(v) => {
            if v.as_str() == Some(user.as_str()) {
                Ok(())
            } else {
                Err((
                    StatusCode::FORBIDDEN,
                    Json(ErrorResponse {
                        error: format!(
                            "rls: cannot insert record with owner field value {} as user {user}",
                            v.as_str().unwrap_or("<non-string>"),
                        ),
                    }),
                ))
            }
        }
    }
}

/// Convenience: true when the caller holds Admin role.
pub fn is_admin(claims: &Claims) -> bool {
    claims.role == Role::Admin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Role;
    use veda_query::filter::ast::{FilterExpr, FilterValue};

    fn alice_claims() -> Claims {
        Claims { sub: "alice".to_owned(), role: Role::Editor, exp: u64::MAX }
    }

    fn admin_claims() -> Claims {
        Claims { sub: "admin".to_owned(), role: Role::Admin, exp: u64::MAX }
    }

    #[test]
    fn inject_no_user_filter_non_admin() {
        let f = inject_owner_filter(None, "owner", "alice", false);
        assert_eq!(
            f,
            Some(FilterExpr::Eq {
                field: "owner".to_owned(),
                value: FilterValue::String("alice".to_owned()),
            })
        );
    }

    #[test]
    fn inject_with_user_filter_non_admin() {
        let user_f = FilterExpr::Eq {
            field: "status".to_owned(),
            value: FilterValue::String("active".to_owned()),
        };
        let f = inject_owner_filter(Some(user_f.clone()), "owner", "alice", false);
        assert_eq!(
            f,
            Some(FilterExpr::And(
                Box::new(user_f),
                Box::new(FilterExpr::Eq {
                    field: "owner".to_owned(),
                    value: FilterValue::String("alice".to_owned()),
                }),
            ))
        );
    }

    #[test]
    fn inject_admin_passthrough() {
        let user_f = FilterExpr::Eq {
            field: "x".to_owned(),
            value: FilterValue::String("y".to_owned()),
        };
        let f = inject_owner_filter(Some(user_f.clone()), "owner", "admin", true);
        assert_eq!(f, Some(user_f));
        let f2 = inject_owner_filter(None, "owner", "admin", true);
        assert_eq!(f2, None);
    }

    #[test]
    fn enforce_sets_missing_owner() {
        let mut rec = serde_json::Map::new();
        enforce_owner_on_insert(&mut rec, "owner", &alice_claims()).unwrap();
        assert_eq!(rec.get("owner"), Some(&serde_json::Value::String("alice".to_owned())));
    }

    #[test]
    fn enforce_allows_own_owner() {
        let mut rec = serde_json::Map::new();
        rec.insert("owner".to_owned(), serde_json::Value::String("alice".to_owned()));
        enforce_owner_on_insert(&mut rec, "owner", &alice_claims()).unwrap();
    }

    #[test]
    fn enforce_rejects_different_owner() {
        let mut rec = serde_json::Map::new();
        rec.insert("owner".to_owned(), serde_json::Value::String("bob".to_owned()));
        let result = enforce_owner_on_insert(&mut rec, "owner", &alice_claims());
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn enforce_admin_allows_any_owner() {
        let mut rec = serde_json::Map::new();
        rec.insert("owner".to_owned(), serde_json::Value::String("alice".to_owned()));
        enforce_owner_on_insert(&mut rec, "owner", &admin_claims()).unwrap();
    }
}

/// Serialize a FilterExpr back to the filter DSL string that FilterParser understands.
/// Only needs to handle the nodes produced by inject_owner_filter (Eq + And wrapping),
/// plus the passthrough of user-supplied nodes for completeness.
pub fn filter_expr_to_string(expr: &FilterExpr) -> String {
    match expr {
        FilterExpr::Eq { field, value } => {
            let v = fmt_value(value);
            format!("{field} = {v}")
        }
        FilterExpr::Ne { field, value } => format!("{field} != {}", fmt_value(value)),
        FilterExpr::Gt { field, value } => format!("{field} > {}", fmt_value(value)),
        FilterExpr::Gte { field, value } => format!("{field} >= {}", fmt_value(value)),
        FilterExpr::Lt { field, value } => format!("{field} < {}", fmt_value(value)),
        FilterExpr::Lte { field, value } => format!("{field} <= {}", fmt_value(value)),
        FilterExpr::Between { field, low, high } => {
            format!("{field} BETWEEN {} AND {}", fmt_value(low), fmt_value(high))
        }
        FilterExpr::In { field, values } => {
            let vs: Vec<String> = values.iter().map(fmt_value).collect();
            format!("{field} IN ({})", vs.join(", "))
        }
        FilterExpr::Contains { field, value } => format!("{field} CONTAINS '{value}'"),
        FilterExpr::StartsWith { field, value } => format!("{field} STARTS_WITH '{value}'"),
        FilterExpr::IsNull { field } => format!("{field} IS NULL"),
        FilterExpr::IsNotNull { field } => format!("{field} IS NOT NULL"),
        FilterExpr::And(l, r) => {
            format!("({}) AND ({})", filter_expr_to_string(l), filter_expr_to_string(r))
        }
        FilterExpr::Or(l, r) => {
            format!("({}) OR ({})", filter_expr_to_string(l), filter_expr_to_string(r))
        }
        FilterExpr::Not(inner) => format!("NOT ({})", filter_expr_to_string(inner)),
    }
}

fn fmt_value(v: &FilterValue) -> String {
    match v {
        FilterValue::String(s) => format!("'{s}'"),
        FilterValue::Int(i) => i.to_string(),
        FilterValue::Float(f) => f.to_string(),
        FilterValue::Bool(b) => b.to_string(),
    }
}

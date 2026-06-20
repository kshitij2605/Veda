/// AST nodes for filter expressions.
///
/// The filter language supports comparison operators, range checks, set
/// membership, string operations, null checks, and logical combinators.

#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    // Comparison operators
    Eq {
        field: String,
        value: FilterValue,
    },
    Ne {
        field: String,
        value: FilterValue,
    },
    Gt {
        field: String,
        value: FilterValue,
    },
    Gte {
        field: String,
        value: FilterValue,
    },
    Lt {
        field: String,
        value: FilterValue,
    },
    Lte {
        field: String,
        value: FilterValue,
    },

    // Range
    Between {
        field: String,
        low: FilterValue,
        high: FilterValue,
    },

    // Set membership
    In {
        field: String,
        values: Vec<FilterValue>,
    },

    // String operations
    Contains {
        field: String,
        value: String,
    },
    StartsWith {
        field: String,
        value: String,
    },

    // Null checks
    IsNull {
        field: String,
    },
    IsNotNull {
        field: String,
    },

    // Logical operators
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
    Not(Box<FilterExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl FilterValue {
    /// Convert the value to f64, for use with range index queries.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FilterValue::Int(i) => Some(*i as f64),
            FilterValue::Float(f) => Some(*f),
            FilterValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            FilterValue::String(_) => None,
        }
    }

    /// Convert the value to a string representation, for use with inverted/bitmap indexes.
    pub fn as_str_repr(&self) -> String {
        match self {
            FilterValue::String(s) => s.clone(),
            FilterValue::Int(i) => i.to_string(),
            FilterValue::Float(f) => f.to_string(),
            FilterValue::Bool(b) => b.to_string(),
        }
    }
}

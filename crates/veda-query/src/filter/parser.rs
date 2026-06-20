use veda_core::error::{Result, VedaError};

use super::ast::{FilterExpr, FilterValue};

// ── Token types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    // Identifiers and literals
    Ident(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),

    // Keywords (stored normalised to uppercase)
    And,
    Or,
    Not,
    Between,
    In,
    Contains,
    StartsWith,
    /// SQL-style `LIKE 'pattern'`. The parser translates this into
    /// `Contains` (when the pattern is `%X%`) or `StartsWith` (`X%`).
    /// GAP-1 fix (2026-05-21).
    Like,
    Is,
    Null,

    // Operators
    Eq,  // =
    Ne,  // !=
    Gt,  // >
    Gte, // >=
    Lt,  // <
    Lte, // <=

    // Punctuation
    LParen,
    RParen,
    Comma,
    Dot,

    Eof,
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Single- or double-quoted string literal. UX-2 fix
        // (2026-05-21): the lexer previously rejected double
        // quotes with "unexpected character: '\"'", surprising
        // users who write `field = "foo"`.
        if ch == '\'' || ch == '"' {
            let quote = ch;
            i += 1;
            let start = i;
            while i < len && chars[i] != quote {
                i += 1;
            }
            if i >= len {
                return Err(VedaError::Query("unterminated string literal".into()));
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push(Token::StringLit(s));
            i += 1; // skip closing quote
            continue;
        }

        // Operators
        if ch == '=' {
            tokens.push(Token::Eq);
            i += 1;
            continue;
        }
        if ch == '!' && i + 1 < len && chars[i + 1] == '=' {
            tokens.push(Token::Ne);
            i += 2;
            continue;
        }
        if ch == '>' {
            if i + 1 < len && chars[i + 1] == '=' {
                tokens.push(Token::Gte);
                i += 2;
            } else {
                tokens.push(Token::Gt);
                i += 1;
            }
            continue;
        }
        if ch == '<' {
            if i + 1 < len && chars[i + 1] == '=' {
                tokens.push(Token::Lte);
                i += 2;
            } else {
                tokens.push(Token::Lt);
                i += 1;
            }
            continue;
        }

        // Punctuation
        if ch == '(' {
            tokens.push(Token::LParen);
            i += 1;
            continue;
        }
        if ch == ')' {
            tokens.push(Token::RParen);
            i += 1;
            continue;
        }
        if ch == ',' {
            tokens.push(Token::Comma);
            i += 1;
            continue;
        }
        if ch == '.' {
            tokens.push(Token::Dot);
            i += 1;
            continue;
        }

        // Numbers (including negative)
        if ch.is_ascii_digit() || (ch == '-' && i + 1 < len && chars[i + 1].is_ascii_digit()) {
            let start = i;
            if ch == '-' {
                i += 1;
            }
            while i < len && chars[i].is_ascii_digit() {
                i += 1;
            }
            let is_float =
                i < len && chars[i] == '.' && i + 1 < len && chars[i + 1].is_ascii_digit();
            if is_float {
                i += 1; // skip dot
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let f: f64 = s
                    .parse()
                    .map_err(|_| VedaError::Query(format!("invalid float: {s}")))?;
                tokens.push(Token::FloatLit(f));
            } else {
                let s: String = chars[start..i].iter().collect();
                let n: i64 = s
                    .parse()
                    .map_err(|_| VedaError::Query(format!("invalid integer: {s}")))?;
                tokens.push(Token::IntLit(n));
            }
            continue;
        }

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let upper = word.to_ascii_uppercase();
            let token = match upper.as_str() {
                "AND" => Token::And,
                "OR" => Token::Or,
                "NOT" => Token::Not,
                "BETWEEN" => Token::Between,
                "IN" => Token::In,
                "CONTAINS" => Token::Contains,
                "STARTS_WITH" => Token::StartsWith,
                // GAP-1 fix (2026-05-21): SQL-style LIKE. ILIKE is
                // accepted as an alias for now (true case-insensitive
                // matching across the index would need lowering the
                // stored values too — tracked as a follow-up).
                "LIKE" | "ILIKE" => Token::Like,
                "IS" => Token::Is,
                "NULL" => Token::Null,
                "TRUE" => Token::BoolLit(true),
                "FALSE" => Token::BoolLit(false),
                _ => Token::Ident(word),
            };
            tokens.push(token);
            continue;
        }

        return Err(VedaError::Query(format!("unexpected character: '{ch}'")));
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

// ── Parser ─────────────────────────────────────────────────────────────────

/// Recursive-descent parser for filter expressions.
pub struct FilterParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl FilterParser {
    /// Parse a filter expression string into an AST.
    pub fn parse(input: &str) -> Result<FilterExpr> {
        let tokens = tokenize(input)?;
        let mut parser = FilterParser { tokens, pos: 0 };
        let expr = parser.parse_or()?;
        if !parser.at_eof() {
            return Err(VedaError::Query(format!(
                "unexpected token after expression: {:?}",
                parser.peek()
            )));
        }
        Ok(expr)
    }

    // ── Helpers ────────────────────────────────────────────────────────

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(VedaError::Query(format!(
                "expected {expected:?}, got {tok:?}"
            )))
        }
    }

    // ── Grammar rules ──────────────────────────────────────────────────

    /// or_expr = and_expr ("OR" and_expr)*
    fn parse_or(&mut self) -> Result<FilterExpr> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = FilterExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// and_expr = not_expr ("AND" not_expr)*
    fn parse_and(&mut self) -> Result<FilterExpr> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Token::And) {
            self.advance();
            let right = self.parse_not()?;
            left = FilterExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// not_expr = "NOT" not_expr | primary
    fn parse_not(&mut self) -> Result<FilterExpr> {
        if matches!(self.peek(), Token::Not) {
            self.advance();
            let inner = self.parse_not()?;
            Ok(FilterExpr::Not(Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    /// primary = comparison | "(" expr ")"
    fn parse_primary(&mut self) -> Result<FilterExpr> {
        if matches!(self.peek(), Token::LParen) {
            self.advance();
            let expr = self.parse_or()?;
            self.expect(&Token::RParen)?;
            return Ok(expr);
        }
        self.parse_comparison()
    }

    /// Parse a field name, which may be a dotted path (e.g. `metadata.color`).
    fn parse_field(&mut self) -> Result<String> {
        let tok = self.advance();
        let first = match tok {
            Token::Ident(s) => s,
            other => {
                return Err(VedaError::Query(format!(
                    "expected field name, got {other:?}"
                )));
            }
        };
        let mut field = first;
        while matches!(self.peek(), Token::Dot) {
            self.advance(); // consume '.'
            let tok = self.advance();
            match tok {
                Token::Ident(s) => {
                    field.push('.');
                    field.push_str(&s);
                }
                other => {
                    return Err(VedaError::Query(format!(
                        "expected identifier after '.', got {other:?}"
                    )));
                }
            }
        }
        Ok(field)
    }

    /// Parse a value literal.
    fn parse_value(&mut self) -> Result<FilterValue> {
        let tok = self.advance();
        match tok {
            Token::StringLit(s) => Ok(FilterValue::String(s)),
            Token::IntLit(n) => Ok(FilterValue::Int(n)),
            Token::FloatLit(f) => Ok(FilterValue::Float(f)),
            Token::BoolLit(b) => Ok(FilterValue::Bool(b)),
            other => Err(VedaError::Query(format!("expected value, got {other:?}"))),
        }
    }

    /// comparison = field operator value
    ///            | field "BETWEEN" value "AND" value
    ///            | field "IN" "(" value ("," value)* ")"
    ///            | field "CONTAINS" string_value
    ///            | field "STARTS_WITH" string_value
    ///            | field "IS" "NULL"
    ///            | field "IS" "NOT" "NULL"
    fn parse_comparison(&mut self) -> Result<FilterExpr> {
        let field = self.parse_field()?;
        let op = self.peek().clone();

        match op {
            Token::Eq => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Eq { field, value })
            }
            Token::Ne => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Ne { field, value })
            }
            Token::Gt => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Gt { field, value })
            }
            Token::Gte => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Gte { field, value })
            }
            Token::Lt => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Lt { field, value })
            }
            Token::Lte => {
                self.advance();
                let value = self.parse_value()?;
                Ok(FilterExpr::Lte { field, value })
            }
            Token::Between => {
                self.advance();
                let low = self.parse_value()?;
                self.expect(&Token::And)?;
                let high = self.parse_value()?;
                Ok(FilterExpr::Between { field, low, high })
            }
            Token::In => {
                self.advance();
                self.expect(&Token::LParen)?;
                let mut values = vec![self.parse_value()?];
                while matches!(self.peek(), Token::Comma) {
                    self.advance();
                    values.push(self.parse_value()?);
                }
                self.expect(&Token::RParen)?;
                Ok(FilterExpr::In { field, values })
            }
            Token::Contains => {
                self.advance();
                let tok = self.advance();
                match tok {
                    Token::StringLit(s) => Ok(FilterExpr::Contains { field, value: s }),
                    other => Err(VedaError::Query(format!(
                        "CONTAINS expects string literal, got {other:?}"
                    ))),
                }
            }
            Token::StartsWith => {
                self.advance();
                let tok = self.advance();
                match tok {
                    Token::StringLit(s) => Ok(FilterExpr::StartsWith { field, value: s }),
                    other => Err(VedaError::Query(format!(
                        "STARTS_WITH expects string literal, got {other:?}"
                    ))),
                }
            }
            Token::Like => {
                // GAP-1 fix (2026-05-21): translate SQL LIKE patterns
                // into the existing string-op variants.
                //   '%X%' → Contains
                //   'X%'  → StartsWith
                //   'X'   → Eq (literal equality, no wildcards)
                //   '%X'  → ends-with — not yet supported; emit a
                //          clear error instead of silently doing
                //          the wrong thing.
                self.advance();
                let tok = self.advance();
                let pattern = match tok {
                    Token::StringLit(s) => s,
                    other => {
                        return Err(VedaError::Query(format!(
                            "LIKE expects string literal, got {other:?}"
                        )));
                    }
                };
                let starts = pattern.starts_with('%');
                let ends = pattern.ends_with('%') && pattern.len() > 1;
                let body: String = pattern.trim_matches('%').to_owned();
                match (starts, ends) {
                    (true, true) | (false, false) if body.contains('%') => {
                        Err(VedaError::Query(format!(
                            "LIKE pattern '{pattern}' has '%' in the middle — only leading and/or trailing '%' are supported"
                        )))
                    }
                    (true, true) => Ok(FilterExpr::Contains { field, value: body }),
                    (false, true) => Ok(FilterExpr::StartsWith { field, value: body }),
                    (true, false) => Err(VedaError::Query(format!(
                        "LIKE pattern '{pattern}' (ends-with) is not yet supported; use CONTAINS or '%{body}%'"
                    ))),
                    (false, false) => Ok(FilterExpr::Eq {
                        field,
                        value: FilterValue::String(pattern),
                    }),
                }
            }
            Token::Is => {
                self.advance();
                if matches!(self.peek(), Token::Not) {
                    self.advance();
                    self.expect(&Token::Null)?;
                    Ok(FilterExpr::IsNotNull { field })
                } else {
                    self.expect(&Token::Null)?;
                    Ok(FilterExpr::IsNull { field })
                }
            }
            other => Err(VedaError::Query(format!(
                "expected operator after field '{field}', got {other:?}"
            ))),
        }
    }
}

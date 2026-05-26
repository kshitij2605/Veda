use std::time::{SystemTime, UNIX_EPOCH};

/// Parse a `?at=` value into a u128 commit_ts.
///
/// Accepted formats:
/// - Raw decimal u128 (existing behaviour)
/// - `now` — current wall-clock as HLC-shaped ts
/// - `now-<N>h` / `now-<N>m` / `now-<N>s` — relative offset
///
/// The HLC encoding is `(physical_ns << 32) | logical`. For relative
/// timestamps we set `logical = 0` since we only need approximate
/// snapshot selection, not ordering guarantees.
pub fn parse_read_ts(s: &str) -> Result<u128, String> {
    let s = s.trim();

    if let Ok(v) = s.parse::<u128>() {
        return Ok(v);
    }

    if !s.starts_with("now") {
        return Err(format!(
            "invalid at='{s}': expected decimal u128 or relative \
             expression (now, now-1h, now-30m, now-5s)"
        ));
    }

    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    if s == "now" {
        return Ok((now_ns as u128) << 32);
    }

    let rest = s.strip_prefix("now-").ok_or_else(|| {
        format!("invalid at='{s}': expected now-<N>h, now-<N>m, or now-<N>s")
    })?;

    let (num_str, unit_ns): (&str, u64) = if let Some(n) = rest.strip_suffix('h') {
        (n, 3_600_000_000_000)
    } else if let Some(n) = rest.strip_suffix('m') {
        (n, 60_000_000_000)
    } else if let Some(n) = rest.strip_suffix('s') {
        (n, 1_000_000_000)
    } else {
        return Err(format!(
            "invalid at='{s}': unit must be h, m, or s (e.g. now-1h, now-30m, now-5s)"
        ));
    };

    let n: u64 = num_str.parse().map_err(|_| {
        format!("invalid at='{s}': '{num_str}' is not a valid integer")
    })?;

    let offset_ns = n.saturating_mul(unit_ns);
    let target_ns = now_ns.saturating_sub(offset_ns);
    Ok((target_ns as u128) << 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_u128_passthrough() {
        let v = parse_read_ts("12345678901234567890").unwrap();
        assert_eq!(v, 12345678901234567890u128);
    }

    #[test]
    fn now_returns_recent_ts() {
        let v = parse_read_ts("now").unwrap();
        let physical = (v >> 32) as u64;
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        assert!((wall_ns - physical) < 1_000_000_000, "should be within 1s of wall clock");
    }

    #[test]
    fn now_minus_1h() {
        let v = parse_read_ts("now-1h").unwrap();
        let physical = (v >> 32) as u64;
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let diff = wall_ns - physical;
        let one_hour_ns = 3_600_000_000_000u64;
        assert!(diff > one_hour_ns - 2_000_000_000 && diff < one_hour_ns + 2_000_000_000,
            "should be ~1h ago, got diff={diff}ns");
    }

    #[test]
    fn now_minus_30m() {
        let v = parse_read_ts("now-30m").unwrap();
        let physical = (v >> 32) as u64;
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let diff = wall_ns - physical;
        let thirty_min_ns = 1_800_000_000_000u64;
        assert!(diff > thirty_min_ns - 2_000_000_000 && diff < thirty_min_ns + 2_000_000_000);
    }

    #[test]
    fn now_minus_5s() {
        let v = parse_read_ts("now-5s").unwrap();
        let physical = (v >> 32) as u64;
        let wall_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let diff = wall_ns - physical;
        let five_sec_ns = 5_000_000_000u64;
        assert!(diff > five_sec_ns - 2_000_000_000 && diff < five_sec_ns + 2_000_000_000);
    }

    #[test]
    fn invalid_format_rejected() {
        assert!(parse_read_ts("yesterday").is_err());
        assert!(parse_read_ts("now-abc").is_err());
        assert!(parse_read_ts("now-1d").is_err());
        assert!(parse_read_ts("now+1h").is_err());
    }
}

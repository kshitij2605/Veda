//! SSRF URL guard for the web-fetch entry points.
//!
//! Every Veda surface that hands a user-provided URL to crawl4ai —
//! `POST /v1/collections/{name}/ingest-url`, the Web connector `/run`
//! action, and the agentic `fetch_url` tool — first runs the target URL
//! through [`check_url_allowed`]. A blocked URL is rejected *before* any
//! fetch, so a crafted request cannot make the server reach internal
//! infrastructure (cloud metadata, RFC1918 hosts, loopback, etc.).
//!
//! The check is **default-secure**: only `http`/`https` are permitted and
//! any host that resolves to a loopback / private / link-local / metadata
//! / unique-local-IPv6 / reserved address is rejected. Because we inspect
//! the *resolved* IP(s), DNS-rebinding-to-internal is blocked too.
//!
//! crawl4ai itself should run as a separate, ideally network-isolated,
//! container; this guard is defense-in-depth at the Veda layer.
//!
//! Optional env knobs (parsed by [`web_policy`]):
//!   - `VEDA_WEB_ALLOWLIST` — comma-separated host suffixes. When set, the
//!     target host MUST match one of them (else rejected).
//!   - `VEDA_WEB_BLOCKLIST` — comma-separated host suffixes always rejected.
//!   - `VEDA_WEB_ALLOW_PRIVATE=1` — escape hatch permitting private /
//!     internal addresses (for trusted self-hosted crawling). Default off.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};

/// Reason a URL was rejected by [`check_url_allowed`].
#[derive(Debug, PartialEq, Eq)]
pub enum UrlGuardError {
    /// URL could not be parsed.
    Malformed(String),
    /// Scheme other than http/https.
    BadScheme(String),
    /// URL had no host component.
    NoHost,
    /// DNS resolution of the host failed.
    Unresolvable(String),
    /// Host resolved to (or was) a private/internal/reserved IP.
    PrivateAddress(String),
    /// An allowlist is set and the host matched none of its suffixes.
    NotAllowlisted(String),
    /// Host matched a blocklist suffix.
    Blocklisted(String),
}

impl UrlGuardError {
    /// Short machine code for the REST/tool error envelope.
    pub fn code(&self) -> &'static str {
        match self {
            UrlGuardError::Malformed(_) => "URL_MALFORMED",
            UrlGuardError::BadScheme(_) => "URL_BAD_SCHEME",
            UrlGuardError::NoHost => "URL_NO_HOST",
            UrlGuardError::Unresolvable(_) => "URL_UNRESOLVABLE",
            UrlGuardError::PrivateAddress(_) => "URL_BLOCKED_PRIVATE",
            UrlGuardError::NotAllowlisted(_) => "URL_NOT_ALLOWLISTED",
            UrlGuardError::Blocklisted(_) => "URL_BLOCKLISTED",
        }
    }
}

impl std::fmt::Display for UrlGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UrlGuardError::Malformed(u) => write!(f, "URL could not be parsed: {u}"),
            UrlGuardError::BadScheme(s) => {
                write!(f, "URL scheme '{s}' is not allowed (only http/https)")
            }
            UrlGuardError::NoHost => write!(f, "URL has no host"),
            UrlGuardError::Unresolvable(h) => write!(f, "host '{h}' could not be resolved"),
            UrlGuardError::PrivateAddress(h) => write!(
                f,
                "host '{h}' resolves to a private/internal/reserved address and is blocked"
            ),
            UrlGuardError::NotAllowlisted(h) => {
                write!(f, "host '{h}' is not in the configured allowlist")
            }
            UrlGuardError::Blocklisted(h) => write!(f, "host '{h}' is blocklisted"),
        }
    }
}

/// Parsed env-driven web access policy. Built once per call from the
/// process environment so tests can drive it deterministically.
#[derive(Debug, Default, Clone)]
pub struct WebPolicy {
    /// Host suffixes; when non-empty the host must match one.
    pub allowlist: Vec<String>,
    /// Host suffixes always rejected.
    pub blocklist: Vec<String>,
    /// When true, private/internal addresses are permitted.
    pub allow_private: bool,
}

/// Read [`WebPolicy`] from the environment. Small, pure-ish helper so the
/// list/escape-hatch parsing is unit-testable in isolation.
pub fn web_policy() -> WebPolicy {
    WebPolicy {
        allowlist: parse_suffix_list(std::env::var("VEDA_WEB_ALLOWLIST").ok().as_deref()),
        blocklist: parse_suffix_list(std::env::var("VEDA_WEB_BLOCKLIST").ok().as_deref()),
        allow_private: matches!(
            std::env::var("VEDA_WEB_ALLOW_PRIVATE").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        ),
    }
}

/// Split a comma-separated suffix list, lowercasing and trimming, dropping
/// empties. A leading dot is stripped so `.example.com` == `example.com`.
fn parse_suffix_list(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or("")
        .split(',')
        .map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// True when `host` equals `suffix` or ends with `.suffix` (case-insensitive).
fn host_matches_suffix(host: &str, suffix: &str) -> bool {
    let host = host.to_ascii_lowercase();
    host == *suffix || host.ends_with(&format!(".{suffix}"))
}

/// Validate a user-supplied URL before it is fetched. Default-secure:
/// rejects non-http(s) schemes and any host that resolves to a
/// private/internal/reserved address, honouring the env-driven
/// [`WebPolicy`] (allow/block lists + the allow-private escape hatch).
pub fn check_url_allowed(url: &str) -> Result<(), UrlGuardError> {
    check_url_with_policy(url, &web_policy())
}

/// [`check_url_allowed`] with an explicit policy (lets tests avoid env).
pub fn check_url_with_policy(url: &str, policy: &WebPolicy) -> Result<(), UrlGuardError> {
    let parsed =
        url::Url::parse(url).map_err(|_| UrlGuardError::Malformed(url.to_owned()))?;

    // 1. Scheme: only http/https.
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(UrlGuardError::BadScheme(scheme.to_owned()));
    }

    // 2. Host must be present.
    let host = parsed.host_str().ok_or(UrlGuardError::NoHost)?;
    // url percent-decodes/normalises; strip IPv6 brackets for matching.
    let host_bare = host.trim_start_matches('[').trim_end_matches(']');
    let host_lc = host_bare.to_ascii_lowercase();

    // 3. Blocklist always wins.
    if policy
        .blocklist
        .iter()
        .any(|s| host_matches_suffix(&host_lc, s))
    {
        return Err(UrlGuardError::Blocklisted(host_bare.to_owned()));
    }

    // 4. Allowlist (when set) is mandatory.
    if !policy.allowlist.is_empty()
        && !policy
            .allowlist
            .iter()
            .any(|s| host_matches_suffix(&host_lc, s))
    {
        return Err(UrlGuardError::NotAllowlisted(host_bare.to_owned()));
    }

    // 5. Resolve the host and check every resolved address. A bare IP
    //    literal resolves to itself. Use the URL's port (or a default) so
    //    `to_socket_addrs` can run; the port is irrelevant to the IP check.
    let port = parsed.port_or_known_default().unwrap_or(80);
    let addrs: Vec<IpAddr> = (host_bare, port)
        .to_socket_addrs()
        .map_err(|_| UrlGuardError::Unresolvable(host_bare.to_owned()))?
        .map(|sa| sa.ip())
        .collect();
    if addrs.is_empty() {
        return Err(UrlGuardError::Unresolvable(host_bare.to_owned()));
    }

    if !policy.allow_private {
        for ip in &addrs {
            if is_blocked_ip(ip) {
                return Err(UrlGuardError::PrivateAddress(host_bare.to_owned()));
            }
        }
    }

    Ok(())
}

/// True when `ip` is one we must never let the server reach: loopback,
/// RFC1918 private, link-local (incl. `169.254.169.254` metadata),
/// unique-local IPv6, unspecified, multicast, broadcast, or a reserved
/// test/CGNAT range. Composed from STABLE std methods (no nightly
/// `is_global`).
fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

fn is_blocked_v4(ip: &Ipv4Addr) -> bool {
    if ip.is_loopback()        // 127.0.0.0/8
        || ip.is_private()      // 10/8, 172.16/12, 192.168/16
        || ip.is_link_local()  // 169.254.0.0/16 (incl. cloud metadata)
        || ip.is_unspecified() // 0.0.0.0
        || ip.is_multicast()   // 224.0.0.0/4
        || ip.is_broadcast()
    // 255.255.255.255
    {
        return true;
    }
    let o = ip.octets();
    // 0.0.0.0/8 "this network"
    if o[0] == 0 {
        return true;
    }
    // 100.64.0.0/10 carrier-grade NAT
    if o[0] == 100 && (64..=127).contains(&o[1]) {
        return true;
    }
    // 192.0.0.0/24 IETF protocol assignments, 192.0.2.0/24 TEST-NET-1
    if o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2) {
        return true;
    }
    // 198.18.0.0/15 benchmarking
    if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
        return true;
    }
    // 198.51.100.0/24 TEST-NET-2
    if o[0] == 198 && o[1] == 51 && o[2] == 100 {
        return true;
    }
    // 203.0.113.0/24 TEST-NET-3
    if o[0] == 203 && o[1] == 0 && o[2] == 113 {
        return true;
    }
    // 240.0.0.0/4 reserved (excl. broadcast already handled)
    if o[0] >= 240 {
        return true;
    }
    false
}

fn is_blocked_v6(ip: &Ipv6Addr) -> bool {
    if ip.is_loopback()        // ::1
        || ip.is_unspecified() // ::
        || ip.is_multicast()
    // ff00::/8
    {
        return true;
    }
    let seg = ip.segments();
    // fc00::/7 unique-local
    if (seg[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    // fe80::/10 link-local
    if (seg[0] & 0xffc0) == 0xfe80 {
        return true;
    }
    // IPv4-mapped (::ffff:0:0/96) and IPv4-compatible: re-check the
    // embedded v4 against the v4 rules so `::ffff:127.0.0.1` is caught.
    if let Some(v4) = ip.to_ipv4() {
        return is_blocked_v4(&v4);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes env-mutating tests (allowlist / allow-private) so they
    /// don't race on the process-global env, mirroring the crawl-env locks
    /// in the integration tests.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn open() -> WebPolicy {
        WebPolicy::default()
    }

    // ── scheme (no DNS) ──────────────────────────────────────────────────
    #[test]
    fn blocks_file_scheme() {
        let e = check_url_with_policy("file:///etc/passwd", &open()).unwrap_err();
        assert!(matches!(e, UrlGuardError::BadScheme(_)), "{e:?}");
    }

    #[test]
    fn blocks_ftp_scheme() {
        let e = check_url_with_policy("ftp://x/", &open()).unwrap_err();
        assert!(matches!(e, UrlGuardError::BadScheme(_)), "{e:?}");
    }

    // ── private / loopback / metadata IP literals (no DNS) ───────────────
    #[test]
    fn blocks_loopback_v4_literal() {
        assert!(matches!(
            check_url_with_policy("http://127.0.0.1/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_rfc1918_10() {
        assert!(matches!(
            check_url_with_policy("http://10.0.0.1/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_rfc1918_192_168() {
        assert!(matches!(
            check_url_with_policy("http://192.168.1.1/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_cloud_metadata_link_local() {
        assert!(matches!(
            check_url_with_policy("http://169.254.169.254/latest/meta-data/", &open())
                .unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_loopback_v6_literal() {
        assert!(matches!(
            check_url_with_policy("http://[::1]/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_ipv4_mapped_loopback_v6() {
        assert!(matches!(
            check_url_with_policy("http://[::ffff:127.0.0.1]/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_unique_local_v6() {
        assert!(matches!(
            check_url_with_policy("http://[fc00::1]/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    #[test]
    fn blocks_cgnat() {
        assert!(matches!(
            check_url_with_policy("http://100.64.0.1/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }

    // ── allowed public IP literal (no DNS) ───────────────────────────────
    #[test]
    fn allows_public_ipv4_literal() {
        // 93.184.216.34 (example.com's historic IP) — public, must pass.
        assert!(check_url_with_policy("http://93.184.216.34/", &open()).is_ok());
    }

    #[test]
    fn allows_public_ipv6_literal() {
        // 2606:4700::1 (public Cloudflare) — must pass.
        assert!(check_url_with_policy("http://[2606:4700::1]/", &open()).is_ok());
    }

    // ── allow/block-list + escape hatch (no DNS, IP literals) ────────────
    #[test]
    fn allowlist_rejects_unlisted_public_host() {
        let policy = WebPolicy {
            allowlist: vec!["example.com".into()],
            ..Default::default()
        };
        // A public IP literal not matching the allowlist suffix → rejected.
        assert!(matches!(
            check_url_with_policy("http://93.184.216.34/", &policy).unwrap_err(),
            UrlGuardError::NotAllowlisted(_)
        ));
    }

    #[test]
    fn blocklist_rejects_matching_host() {
        let policy = WebPolicy {
            blocklist: vec!["93.184.216.34".into()],
            ..Default::default()
        };
        assert!(matches!(
            check_url_with_policy("http://93.184.216.34/", &policy).unwrap_err(),
            UrlGuardError::Blocklisted(_)
        ));
    }

    #[test]
    fn allow_private_escape_hatch_permits_private_literal() {
        let policy = WebPolicy {
            allow_private: true,
            ..Default::default()
        };
        assert!(check_url_with_policy("http://10.0.0.1/", &policy).is_ok());
    }

    // ── env helper ───────────────────────────────────────────────────────
    #[test]
    fn web_policy_parses_lists_and_flag_from_env() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized on ENV_LOCK.
        unsafe {
            std::env::set_var("VEDA_WEB_ALLOWLIST", "example.com, .veda.dev");
            std::env::set_var("VEDA_WEB_BLOCKLIST", "evil.test");
            std::env::set_var("VEDA_WEB_ALLOW_PRIVATE", "1");
        }
        let p = web_policy();
        unsafe {
            std::env::remove_var("VEDA_WEB_ALLOWLIST");
            std::env::remove_var("VEDA_WEB_BLOCKLIST");
            std::env::remove_var("VEDA_WEB_ALLOW_PRIVATE");
        }
        assert_eq!(p.allowlist, vec!["example.com", "veda.dev"]);
        assert_eq!(p.blocklist, vec!["evil.test"]);
        assert!(p.allow_private);
    }

    #[test]
    fn host_suffix_match_is_dot_anchored() {
        assert!(host_matches_suffix("example.com", "example.com"));
        assert!(host_matches_suffix("a.example.com", "example.com"));
        // Must not match a mere substring/suffix without a dot boundary.
        assert!(!host_matches_suffix("notexample.com", "example.com"));
        assert!(!host_matches_suffix("example.com.evil.test", "example.com"));
    }

    // ── DNS-dependent: localhost resolves to loopback (needs resolver) ────
    // Kept separate; relies on the host having a working resolver/hosts
    // file. `localhost` is in every hosts file so this is stable offline.
    #[test]
    fn blocks_localhost_hostname() {
        assert!(matches!(
            check_url_with_policy("http://localhost/", &open()).unwrap_err(),
            UrlGuardError::PrivateAddress(_)
        ));
    }
}

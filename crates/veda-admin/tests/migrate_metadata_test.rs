//! Phase E hard-remove: `migrate-metadata` is now a no-op
//! stub. This test just confirms the binary builds and the
//! subcommand prints the documented message.

use std::process::Command;

#[test]
fn migrate_metadata_stub_runs_and_explains() {
    let bin = env!("CARGO_BIN_EXE_veda-admin");
    let out = Command::new(bin)
        .args(["migrate-metadata"])
        .output()
        .expect("spawn veda-admin");
    assert!(
        out.status.success(),
        "migrate-metadata stub should succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("no-op"),
        "expected stub message in stdout, got: {stdout}"
    );
    assert!(
        stdout.contains("Phase E"),
        "expected Phase E reference in stdout, got: {stdout}"
    );
}

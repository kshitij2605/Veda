//! `veda-admin` — operator CLI for offline maintenance.
//!
//! Today the only subcommand is `migrate-metadata`, which
//! rewrites a node's `raft_state.bin` from the v0 layout
//! (records + open_txns embedded in the metadata Raft state
//! machine) to the v1 layout (data-plane fields dropped — they
//! moved to per-shard logs + the Percolator coordinator).
//! Phase E of the FoundationDB-shape rearchitecture; see
//! `.claude/plans/structured-mixing-turing.md`.
//!
//! ## Usage
//!
//! ```bash
//! veda-admin migrate-metadata --dir ./veda-data/node-1/raft
//! ```
//!
//! The tool atomically replaces `raft_state.bin` (via a `.tmp`
//! sibling + rename) and prints a `(records_dropped,
//! txns_dropped)` summary. A `.bak.<unix_ts>` copy of the input
//! is left next to the original so a rollback is possible.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use veda_coordinator::raft_storage::migrate_persisted_blob;

#[derive(Debug, Parser)]
#[command(name = "veda-admin", version, about = "Veda offline admin tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Migrate a raft_state.bin file from the v0 layout to v1.
    MigrateMetadata(MigrateMetadataArgs),
}

#[derive(Debug, Parser)]
struct MigrateMetadataArgs {
    /// Directory holding `raft_state.bin`. Required.
    #[arg(long)]
    dir: PathBuf,
    /// Skip the timestamped backup copy of the original blob.
    #[arg(long, default_value_t = false)]
    no_backup: bool,
    /// Print the would-be result without writing anything.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::MigrateMetadata(args) => migrate_metadata(args),
    }
}

fn migrate_metadata(args: MigrateMetadataArgs) -> Result<()> {
    let path = args.dir.join("raft_state.bin");
    if !path.exists() {
        bail!("no raft_state.bin in {}", args.dir.display());
    }
    let bytes = fs::read(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let (out, recs, txns) = migrate_persisted_blob(&bytes)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("migrate {}", path.display()))?;

    println!(
        "migrate-metadata: would drop {recs} record(s), {txns} open txn(s)"
    );

    if args.dry_run {
        println!("dry-run: not writing");
        return Ok(());
    }

    if !args.no_backup {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let backup = path.with_extension(format!("bin.bak.{ts}"));
        fs::copy(&path, &backup)
            .with_context(|| format!("backup → {}", backup.display()))?;
        println!("backup: {}", backup.display());
    }

    let tmp = path.with_extension("bin.tmp");
    fs::write(&tmp, &out).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| format!("rename → {}", path.display()))?;
    println!("wrote: {}", path.display());
    Ok(())
}

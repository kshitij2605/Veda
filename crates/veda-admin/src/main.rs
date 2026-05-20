//! `veda-admin` — operator CLI for offline maintenance.
//!
//! Phase E hard-remove: the original `migrate-metadata`
//! subcommand stripped v0 fields (`records`, `open_txns`) from
//! a `raft_state.bin` blob. With those fields and the matching
//! `ClusterCommand` variants now removed from the codebase
//! entirely, the in-process migration path is gone — any v0
//! blob must be re-processed with the pre-Phase-E-hard-remove
//! binary before this binary will load it.
//!
//! The CLI is kept as scaffolding for future subcommands.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "veda-admin", version, about = "Veda offline admin tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Phase E hard-remove: stub. Use the pre-Phase-E binary if
    /// you need to migrate a v0 raft_state.bin.
    MigrateMetadata,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::MigrateMetadata => {
            println!(
                "veda-admin: migrate-metadata is a no-op in this release.\n\
                 Phase E hard-removed the v0 records/open_txns fields; nothing\n\
                 to migrate. Use the pre-Phase-E binary if you need to drain a\n\
                 v0 raft_state.bin before upgrading."
            );
            Ok(())
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Mirrors crates/veda-log/build.rs. PROTOC must point at a protoc
    // binary (see veda-coordinator/build.rs for install notes).
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/shard_search.proto"], &["proto"])?;
    Ok(())
}

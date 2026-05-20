fn main() -> Result<(), Box<dyn std::error::Error>> {
    // See veda-coordinator/build.rs for the protoc install notes.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &["proto/log_service.proto", "proto/wan_service.proto"],
            &["proto"],
        )?;
    Ok(())
}

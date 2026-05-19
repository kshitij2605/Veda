fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tonic codegen requires a `protoc` binary. The Dockerfile
    // apt-installs `protobuf-compiler` so production builds find it
    // on PATH. Dev machines need it too — on Debian/Ubuntu:
    //     sudo apt install protobuf-compiler
    // On macOS:
    //     brew install protobuf
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/raft.proto"], &["proto"])?;
    Ok(())
}

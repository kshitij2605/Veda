fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the vendored protoc shipped by `protobuf-src` so this
    // crate builds on machines without `protobuf-compiler`
    // pre-installed (Veda's prod containers, CI runners, etc.).
    // SAFETY: build scripts are single-threaded; no other code can
    // be reading the env concurrently.
    unsafe { std::env::set_var("PROTOC", protobuf_src::protoc()) };
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/raft.proto"], &["proto"])?;
    Ok(())
}

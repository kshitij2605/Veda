# Stage 1: Build Rust binary
FROM rust:1.83-bookworm AS rust-builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build --release --bin veda-server

# Stage 2: Build Niriksha frontend
FROM node:22-bookworm AS node-builder
WORKDIR /build
COPY niriksha/package*.json niriksha/
RUN cd niriksha && npm ci
COPY niriksha/ niriksha/
RUN cd niriksha && npm run build

# Stage 3: Final minimal image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /build/target/release/veda-server /usr/local/bin/veda-server
COPY --from=node-builder /build/niriksha/dist /usr/share/veda/niriksha

# Default data directory
RUN mkdir -p /var/veda/data

ENV VEDA_DATA_DIR=/var/veda/data
ENV VEDA_LISTEN_ADDR=0.0.0.0:6340
ENV RUST_LOG=info

EXPOSE 6340

ENTRYPOINT ["veda-server"]
CMD ["/var/veda/data", "0.0.0.0:6340"]

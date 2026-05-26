# Multi-stage Dockerfile for AI Framework (REQ-16.2)
# Stage 1: Build
FROM rust:1.82-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests first for better caching
COPY Cargo.toml Cargo.toml
COPY crates/ crates/

# Build the release binaries
RUN cargo build --release --bin ai && \
    cargo build --release -p ai-server

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binaries from builder
COPY --from=builder /app/target/release/ai /usr/local/bin/ai

# Copy configuration templates
COPY test_config.toml /app/config/default.toml

# Create non-root user
RUN useradd -m -s /bin/bash aiuser
USER aiuser

# Configuration via environment variables
ENV AI_SERVER__HOST=0.0.0.0
ENV AI_SERVER__PORT=8080
ENV AI_LOG_LEVEL=info
ENV RUST_LOG=ai=info

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["ai"]
CMD ["serve", "--port", "8080"]

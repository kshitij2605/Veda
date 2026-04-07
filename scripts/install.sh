#!/bin/bash
set -e
echo "Installing Veda v0.1.0..."

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is required. Install it from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &>/dev/null; then
    echo "Error: Docker Compose v2 is required."
    exit 1
fi

# Download docker-compose.yml
echo "Starting Veda with Docker Compose..."
docker compose up -d

echo ""
echo "Veda is running!"
echo "  REST API:  http://localhost:6340/v1/"
echo "  Niriksha:  http://localhost:6340/niriksha/"
echo "  MinIO:     http://localhost:9001/"
echo ""

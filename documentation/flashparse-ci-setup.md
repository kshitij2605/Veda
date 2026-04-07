# FlashParse Docker Image CI Setup

To enable automatic Docker image publishing for FlashParse, add this workflow
to your FlashParse repository at `.github/workflows/docker-publish.yml`:

```yaml
name: Publish Docker Image

on:
  push:
    branches: [main]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/flashparse

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
```

After setting this up, every push to FlashParse's main branch automatically
publishes a new Docker image to `ghcr.io/kshitij2605/flashparse:latest`.

Veda users then get the update with:
```bash
docker compose pull flashparse
docker compose --profile ocr up -d
```

import requests

from .models import (
    CollectionSchema,
    FieldDefinition,
    SearchResponse,
    SearchResult,
)
from .exceptions import (
    CollectionNotFoundError,
    ModalityNotSupportedError,
    ValidationError,
    VedaError,
)


class VedaClient:
    def __init__(self, url: str = "http://localhost:6340", token: str = None):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def login(self, username: str, password: str) -> str:
        """Login and set auth token. Returns token."""
        data = self._request("POST", "/v1/auth/login", json={
            "username": username,
            "password": password,
        })
        token = data["token"]
        self.session.headers["Authorization"] = f"Bearer {token}"
        return token

    def health(self) -> dict:
        """Check cluster health."""
        return self._request("GET", "/v1/admin/health")

    # ── Collections ──────────────────────────────────────────────────────

    def create_collection(self, schema: CollectionSchema) -> dict:
        """Create a new collection."""
        fields = []
        for f in schema.fields:
            field = {"name": f.name, "type": f.type, "filterable": f.filterable}
            if f.dimension is not None:
                field["dimension"] = f.dimension
            if f.metric is not None:
                field["metric"] = f.metric
            fields.append(field)

        return self._request("POST", "/v1/collections", json={
            "name": schema.name,
            "fields": fields,
            "shard_count": schema.shard_count,
            "replication_factor": schema.replication_factor,
        })

    def list_collections(self) -> list[dict]:
        """List all collections."""
        data = self._request("GET", "/v1/collections")
        return data["collections"]

    def describe_collection(self, name: str) -> dict:
        """Get collection details."""
        return self._request("GET", f"/v1/collections/{name}")

    def drop_collection(self, name: str) -> dict:
        """Delete a collection."""
        return self._request("DELETE", f"/v1/collections/{name}")

    # ── Records ──────────────────────────────────────────────────────────

    def insert(self, collection: str, records: list[dict]) -> dict:
        """Insert records into a collection."""
        return self._request("POST", f"/v1/collections/{collection}/insert", json={
            "records": records,
        })

    def get(self, collection: str, record_id: str) -> dict:
        """Get a record by ID."""
        return self._request("GET", f"/v1/collections/{collection}/records/{record_id}")

    def delete(self, collection: str, ids: list[str]) -> dict:
        """Delete records by IDs."""
        return self._request("DELETE", f"/v1/collections/{collection}/records", json={
            "ids": ids,
        })

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        field: str,
        vector: list[float],
        limit: int = 10,
        filter: str = None,
    ) -> SearchResponse:
        """Single-vector search."""
        body = {
            "query": {
                "vector": {"field": field, "values": vector},
            },
            "limit": limit,
        }
        if filter is not None:
            body["filter"] = filter

        data = self._request("POST", f"/v1/collections/{collection}/search", json=body)
        return self._parse_search_response(data)

    def search_multi(
        self,
        collection: str,
        queries: list[dict],
        fusion: str = "rrf",
        limit: int = 10,
        filter: str = None,
    ) -> SearchResponse:
        """Multi-vector fusion search.

        queries: list of dicts with keys "field", "values", and optional "weight".
        """
        body = {
            "query": {
                "multi_vector": queries,
                "fusion": fusion,
            },
            "limit": limit,
        }
        if filter is not None:
            body["filter"] = filter

        data = self._request("POST", f"/v1/collections/{collection}/search", json=body)
        return self._parse_search_response(data)

    # ── Auto-embed ───────────────────────────────────────────────────────

    def embed_insert(
        self,
        collection: str,
        records: list[dict],
        auto_embed: dict,
    ) -> dict:
        """Insert with auto-embedding.

        auto_embed: mapping of target_field -> {"source_field": ..., "model": ...}
        """
        return self._request(
            "POST",
            f"/v1/collections/{collection}/embed-insert",
            json={"records": records, "auto_embed": auto_embed},
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _parse_search_response(self, data: dict) -> SearchResponse:
        results = [
            SearchResult(
                id=r["id"],
                distance=r["distance"],
                fields=r.get("fields", {}),
            )
            for r in data.get("results", [])
        ]
        return SearchResponse(
            results=results,
            count=data.get("count", len(results)),
            took_ms=data.get("took_ms", 0.0),
        )

    def _request(self, method, path, **kwargs):
        """Make HTTP request, handle errors."""
        resp = self.session.request(method, self.url + path, **kwargs)

        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("error", resp.text)
            except (ValueError, KeyError):
                message = resp.text

            if resp.status_code == 404:
                raise CollectionNotFoundError(message, status_code=404)
            if resp.status_code == 422:
                raise ModalityNotSupportedError(message, status_code=422)
            if resp.status_code == 400:
                raise ValidationError(message, status_code=400)
            raise VedaError(message, status_code=resp.status_code)

        return resp.json()

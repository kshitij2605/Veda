import json
from unittest.mock import MagicMock, patch

import pytest

from veda_client import (
    VedaClient,
    CollectionSchema,
    FieldDefinition,
    SearchResponse,
    SearchResult,
    VedaError,
    CollectionNotFoundError,
    ValidationError,
)


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = json.dumps(json_data or {})
    return resp


class TestHealth:
    def test_health(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(200, {"status": "healthy", "version": "0.1.0"})
        client.session.request = MagicMock(return_value=mock_resp)

        result = client.health()

        assert result == {"status": "healthy", "version": "0.1.0"}
        client.session.request.assert_called_once_with(
            "GET", "http://localhost:6340/v1/admin/health"
        )


class TestCreateCollection:
    def test_create_collection(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(201, {"name": "images", "status": "created"})
        client.session.request = MagicMock(return_value=mock_resp)

        schema = CollectionSchema(
            name="images",
            fields=[
                FieldDefinition(name="title", type="string"),
                FieldDefinition(
                    name="embedding",
                    type="vector",
                    dimension=768,
                    metric="cosine",
                ),
            ],
            shard_count=2,
        )
        result = client.create_collection(schema)

        assert result == {"name": "images", "status": "created"}
        call_args = client.session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:6340/v1/collections")
        payload = call_args[1]["json"]
        assert payload["name"] == "images"
        assert payload["shard_count"] == 2
        assert len(payload["fields"]) == 2
        assert payload["fields"][1]["dimension"] == 768
        assert payload["fields"][1]["metric"] == "cosine"


class TestListCollections:
    def test_list_collections(self):
        client = VedaClient("http://localhost:6340")
        collections_data = [
            {"name": "images", "field_count": 3, "shard_count": 1},
            {"name": "docs", "field_count": 2, "shard_count": 2},
        ]
        mock_resp = _mock_response(200, {"collections": collections_data})
        client.session.request = MagicMock(return_value=mock_resp)

        result = client.list_collections()

        assert len(result) == 2
        assert result[0]["name"] == "images"
        assert result[1]["name"] == "docs"


class TestInsert:
    def test_insert(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(201, {"inserted": 2})
        client.session.request = MagicMock(return_value=mock_resp)

        records = [
            {"title": "cat", "embedding": [0.1, 0.2, 0.3]},
            {"title": "dog", "embedding": [0.4, 0.5, 0.6]},
        ]
        result = client.insert("images", records)

        assert result == {"inserted": 2}
        call_args = client.session.request.call_args
        assert call_args[0] == (
            "POST",
            "http://localhost:6340/v1/collections/images/insert",
        )
        payload = call_args[1]["json"]
        assert len(payload["records"]) == 2


class TestSearch:
    def test_search(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(
            200,
            {
                "results": [
                    {"id": "abc-123", "distance": 0.95, "fields": {"title": "cat"}},
                    {"id": "def-456", "distance": 0.87, "fields": {}},
                ],
                "count": 2,
                "took_ms": 12,
            },
        )
        client.session.request = MagicMock(return_value=mock_resp)

        result = client.search("images", "embedding", [0.1, 0.2, 0.3], limit=5)

        assert isinstance(result, SearchResponse)
        assert result.count == 2
        assert result.took_ms == 12
        assert len(result.results) == 2
        assert isinstance(result.results[0], SearchResult)
        assert result.results[0].id == "abc-123"
        assert result.results[0].distance == 0.95
        assert result.results[0].fields == {"title": "cat"}

        call_args = client.session.request.call_args
        payload = call_args[1]["json"]
        assert payload["query"]["vector"]["field"] == "embedding"
        assert payload["query"]["vector"]["values"] == [0.1, 0.2, 0.3]
        assert payload["limit"] == 5

    def test_search_with_filter(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(200, {"results": [], "count": 0, "took_ms": 1})
        client.session.request = MagicMock(return_value=mock_resp)

        client.search("images", "embedding", [0.1], filter="title == 'cat'")

        payload = client.session.request.call_args[1]["json"]
        assert payload["filter"] == "title == 'cat'"


class TestSearchMulti:
    def test_search_multi(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(
            200,
            {
                "results": [{"id": "x-1", "distance": 0.9, "fields": {}}],
                "count": 1,
                "took_ms": 5,
            },
        )
        client.session.request = MagicMock(return_value=mock_resp)

        queries = [
            {"field": "text_emb", "values": [0.1, 0.2], "weight": 0.7},
            {"field": "image_emb", "values": [0.3, 0.4], "weight": 0.3},
        ]
        result = client.search_multi("multimodal", queries, fusion="rrf", limit=5)

        assert isinstance(result, SearchResponse)
        assert result.count == 1
        assert result.results[0].id == "x-1"

        payload = client.session.request.call_args[1]["json"]
        assert payload["query"]["multi_vector"] == queries
        assert payload["query"]["fusion"] == "rrf"
        assert payload["limit"] == 5


class TestErrorHandling:
    def test_404_raises_collection_not_found(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(
            404, {"error": "collection 'missing' not found"}
        )
        client.session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(CollectionNotFoundError) as exc_info:
            client.describe_collection("missing")

        assert exc_info.value.status_code == 404
        assert "collection 'missing' not found" in str(exc_info.value)

    def test_400_raises_validation_error(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(
            400, {"error": "collection name must not be empty"}
        )
        client.session.request = MagicMock(return_value=mock_resp)

        schema = CollectionSchema(name="", fields=[])
        with pytest.raises(ValidationError) as exc_info:
            client.create_collection(schema)

        assert exc_info.value.status_code == 400

    def test_500_raises_veda_error(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(500, {"error": "internal failure"})
        client.session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(VedaError) as exc_info:
            client.health()

        assert exc_info.value.status_code == 500


class TestAuthToken:
    def test_auth_token_set_in_constructor(self):
        client = VedaClient("http://localhost:6340", token="my-jwt-token")
        assert client.session.headers["Authorization"] == "Bearer my-jwt-token"

    def test_no_auth_header_without_token(self):
        client = VedaClient("http://localhost:6340")
        assert "Authorization" not in client.session.headers


class TestLogin:
    def test_login_sets_token(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(200, {"token": "jwt-token-abc"})
        client.session.request = MagicMock(return_value=mock_resp)

        token = client.login("admin", "password123")

        assert token == "jwt-token-abc"
        assert client.session.headers["Authorization"] == "Bearer jwt-token-abc"

        call_args = client.session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:6340/v1/auth/login")
        payload = call_args[1]["json"]
        assert payload["username"] == "admin"
        assert payload["password"] == "password123"

    def test_login_failure(self):
        client = VedaClient("http://localhost:6340")
        mock_resp = _mock_response(401, {"error": "invalid credentials"})
        client.session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(VedaError) as exc_info:
            client.login("bad_user", "wrong_pass")

        assert exc_info.value.status_code == 401

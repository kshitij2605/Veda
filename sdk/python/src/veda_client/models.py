from dataclasses import dataclass
from typing import Optional


@dataclass
class FieldDefinition:
    name: str
    type: str  # "string", "float64", "vector", "blob", etc.
    dimension: Optional[int] = None
    metric: Optional[str] = None  # "cosine", "euclidean", "dot_product"
    filterable: bool = False


@dataclass
class CollectionSchema:
    name: str
    fields: list[FieldDefinition]
    shard_count: int = 1
    replication_factor: int = 1


@dataclass
class SearchResult:
    id: str
    distance: float
    fields: dict


@dataclass
class SearchResponse:
    results: list[SearchResult]
    count: int
    took_ms: float

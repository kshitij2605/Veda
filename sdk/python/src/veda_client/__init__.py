from .client import VedaClient
from .models import CollectionSchema, FieldDefinition, SearchResult, SearchResponse
from .exceptions import (
    VedaError,
    CollectionNotFoundError,
    ModalityNotSupportedError,
    ValidationError,
)

__version__ = "0.1.0"
__all__ = [
    "VedaClient",
    "CollectionSchema",
    "FieldDefinition",
    "SearchResult",
    "SearchResponse",
    "VedaError",
    "CollectionNotFoundError",
    "ModalityNotSupportedError",
    "ValidationError",
]

"""Chunk database model — one embedded slice of a document.

A chunk is the unit of retrieval: a passage of a document's text plus the vector
it embeds to. The parent document's bytes and metadata live elsewhere (blob
storage and the ``documents`` collection); this collection holds the searchable
pieces, one row per chunk.

As with ``document.py`` the shape is declared three ways: a Pydantic model for
application code, ``CHUNKS_INDEXES`` for the lookup/uniqueness constraints, and
``CHUNKS_VALIDATOR`` for the database-level ``$jsonSchema``. The **vector search
index** over ``embedding`` is a separate, Atlas-specific mechanism added in step
5b — it is not one of the ``create_index`` specs here.
"""

from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

COLLECTION_NAME = "chunks"

IndexSpec = tuple[str | list[tuple[str, int]], dict]


class ChunkDocument(BaseModel):
    """Shape of a document in the ``chunks`` collection."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: ObjectId | None = Field(default=None, alias="_id")
    # The parent document (foreign key to documents._id). Every read and the
    # idempotent re-ingest delete filter on this.
    document_id: ObjectId
    # Denormalised owner, copied from the parent. Carried here so a vector search
    # can filter to one tenant without a join back to documents.
    user_id: ObjectId
    # 0-based position of this chunk within its document, in reading order.
    chunk_index: int
    # The chunk's text — what was embedded, and what gets returned at retrieval.
    text: str
    # The embedding vector. Length matches settings.embedding_dimensions (1024 for
    # voyage-4-large); the vector search index (step 5b) is built for that width.
    embedding: list[float]
    created_at: datetime


# The Atlas Vector Search index over `embedding`. Unlike CHUNKS_INDEXES (regular
# btree indexes created with create_index), this goes through the Atlas Search
# index API (create_search_index) — see app.db.mongodb._ensure_vector_index. Only
# Atlas (Cloud, or the Atlas Local Docker image) supports it; plain community
# mongod does not, which is why its creation is best-effort.
VECTOR_INDEX_NAME = "chunks_vector_index"


def build_vector_index_definition(dimensions: int, similarity: str = "cosine") -> dict:
    """The Atlas ``vectorSearch`` index definition over the ``embedding`` field.

    ``similarity`` is cosine: it compares direction, not magnitude, so two chunks
    match on *meaning* regardless of vector length — the right default for text
    embeddings. ``numDimensions`` must equal the embedding width the model emits
    (1024 for voyage-4-large); a mismatch makes every query fail.

    The ``filter`` fields let ``$vectorSearch`` scope a query *inside* the ANN
    search. ``$vectorSearch`` can only pre-filter on fields declared here, so both
    scoping needs must be declared:

    * ``user_id`` — tenant isolation. Without it, scoping to one user would need a
      post-filter that runs after ``limit`` has already been applied, silently
      dropping a user's own results.
    * ``document_id`` — lets a caller restrict a search to a single document
      (``POST /v1/search`` with a ``document_id``). Same reason it must be a
      declared filter field rather than a post-filter.
    """
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dimensions,
                "similarity": similarity,
            },
            {
                "type": "filter",
                "path": "user_id",
            },
            {
                "type": "filter",
                "path": "document_id",
            },
        ]
    }


CHUNKS_INDEXES: list[IndexSpec] = [
    # A document's chunks are replaced wholesale on re-ingest (delete-by-document
    # then insert), and read back in order — both filter on document_id. Making
    # (document_id, chunk_index) unique also guarantees no duplicate positions
    # survive a re-ingest race.
    ([("document_id", 1), ("chunk_index", 1)], {"unique": True}),
    # Tenant scoping for search and cleanup.
    ("user_id", {}),
]

CHUNKS_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "document_id",
            "user_id",
            "chunk_index",
            "text",
            "embedding",
            "created_at",
        ],
        "properties": {
            "document_id": {"bsonType": "objectId"},
            "user_id": {"bsonType": "objectId"},
            "chunk_index": {"bsonType": ["int", "long"]},
            "text": {"bsonType": "string"},
            "embedding": {
                "bsonType": "array",
                # Voyage returns floats; PyMongo stores them as BSON double. Length
                # isn't constrained here so a later Matryoshka switch to 512/256d
                # doesn't require touching the validator (the vector index is where
                # dimension is enforced).
                "items": {"bsonType": "double"},
            },
            "created_at": {"bsonType": "date"},
        },
    }
}

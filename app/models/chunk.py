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

import hashlib
from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

COLLECTION_NAME = "chunks"

IndexSpec = tuple[str | list[tuple[str, int]], dict]


def build_chunk_id(
    user_id: ObjectId | str,
    document_id: ObjectId | str,
    text: str,
    occurrence: int,
) -> str:
    """Deterministic, **position-independent** primary key (``_id``) for a chunk.

    Derived from the chunk's *content*, not its position, so a chunk keeps the same
    id even when earlier edits shift it to a different ``chunk_index``. That is what
    lets the incremental sync recognize an unchanged chunk after an in-place
    document edit and skip re-embedding it. A citation can likewise reference a
    chunk by an id that survives a reprocess as long as the text is unchanged.

    The payload mixes four fields, each load-bearing:

    * ``user_id`` — namespaces one tenant's ids away from another's;
    * ``document_id`` — namespaces one document's chunks from another's;
    * ``content_hash`` — makes the id content-addressed: an edited chunk gets a
      fresh id, an unchanged one keeps its id regardless of where it now sits;
    * ``occurrence`` — disambiguates the rare case of two chunks with *identical*
      text in one document (overlap and un-stripped boilerplate make it real).
      It is the 0-based count of earlier identical-text chunks, so two copies get
      distinct ids and neither is lost. Position is deliberately absent — the only
      thing that renumbers occurrences is adding/removing a *duplicate* of the same
      text, not moving unrelated chunks around.

    SHA-256 throughout (not MD5) for consistency with the document content hash in
    ``storage_service`` and to avoid mixing a second, weaker digest.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = f"{user_id}:{document_id}:{content_hash}:{occurrence}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChunkDocument(BaseModel):
    """Shape of a document in the ``chunks`` collection."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    # Deterministic content-addressed id (see build_chunk_id), not an ObjectId:
    # a hex SHA-256 digest, so re-ingesting identical content reproduces it.
    id: str | None = Field(default=None, alias="_id")
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


# The Atlas Search (lexical/BM25) index over `text`. The *second* leg of hybrid
# search, and a sibling to VECTOR_INDEX_NAME above: same Atlas Search index API,
# but type "search" rather than "vectorSearch". Two separate indexes, not one —
# Atlas has no single index serving both $vectorSearch and $search.
TEXT_INDEX_NAME = "chunks_text_index"


def build_text_index_definition() -> dict:
    """The Atlas Search index definition backing the lexical leg of hybrid search.

    Where the vector index matches on *meaning*, this one matches on *terms*: BM25
    over the analyzed tokens of ``text``, which is what rescues the queries dense
    retrieval is worst at — product names, error codes, version numbers, acronyms.
    An embedding averages a whole 1200-char chunk into 1024 floats and loses the
    fact that a rare token appeared once; an inverted index does not.

    ``dynamic: False`` maps only the fields we name. Deliberate: dynamic mapping
    would also index ``embedding``, ballooning the index with 1024 numbers per
    chunk that no lexical query will ever match.

    ``lucene.standard`` is the analyzer — Unicode word breaks and lowercasing, no
    stemming or stopword removal. Conservative on purpose for a technical corpus,
    where stemming can merge terms that should stay distinct.

    ``user_id`` and ``document_id`` are mapped as ``objectId`` so they can be used
    in a ``compound.filter`` ``equals`` clause *inside* the search. This is the
    same scoping requirement the vector index's ``filter`` fields exist for, and
    for the same reason: a post-``$match`` would run after the leg's ``$limit``
    had already been spent, silently dropping the caller's own results.
    """
    return {
        "mappings": {
            "dynamic": False,
            "fields": {
                "text": {"type": "string", "analyzer": "lucene.standard"},
                "user_id": {"type": "objectId"},
                "document_id": {"type": "objectId"},
            },
        }
    }


CHUNKS_INDEXES: list[IndexSpec] = [
    # A document's chunks are read back in order and reconciled per document — both
    # filter on document_id. NOT unique on (document_id, chunk_index): the
    # content-addressed _id is the real key, and chunk_index is a *mutable* ordering
    # attribute the incremental sync rewrites when a chunk moves (an in-place edit
    # shifts positions). A unique constraint would fight those position updates —
    # uniqueness of a chunk is already guaranteed by its _id. Kept as a plain
    # compound index for ordered reads. (The startup schema check relaxes a
    # pre-existing unique index of the same shape to this — see _ensure_index.)
    ([("document_id", 1), ("chunk_index", 1)], {}),
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

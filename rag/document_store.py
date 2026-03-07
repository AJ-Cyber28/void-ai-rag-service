"""
Shared Haystack PgvectorDocumentStore configuration.
Imported by ingestion scripts (Steps 1 & 2) and retrieval pipelines (Step 4).
"""

import os
import pathlib
from dotenv import load_dotenv
from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

# Load env
_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")


def get_document_store() -> PgvectorDocumentStore:
    conn_string = os.getenv("PG_CONN_STRING")
    if not conn_string:
        raise RuntimeError("PG_CONN_STRING not set in .env.local")

    return PgvectorDocumentStore(
        connection_string=Secret.from_token(conn_string),
        table_name="haystack_documents",
        embedding_dimension=384,
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="hnsw",
    )
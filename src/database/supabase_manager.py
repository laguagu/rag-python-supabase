"""
Supabase database configuration and vector store setup
"""

import logging
import os
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import Client, create_client

logger = logging.getLogger(__name__)


class SupabaseManager:
    """Manages Supabase client and vector operations"""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
            )

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def initialize_vector_store(
        self, table_name: str = "documents"
    ) -> SupabaseVectorStore:
        """Initialize the Supabase vector store"""
        try:
            self.vector_store = SupabaseVectorStore(
                client=self.client,
                embedding=self.embeddings,
                table_name=table_name,
                query_name="match_documents",
            )
            logger.info(f"Vector store initialized with table: {table_name}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call initialize_vector_store() first."
            )

        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict] = None
    ) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call initialize_vector_store() first."
            )

        try:
            results = self.vector_store.similarity_search(
                query=query, k=k, filter=filter
            )
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call initialize_vector_store() first."
            )

        try:
            results = self.vector_store.similarity_search_with_score(query=query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            raise

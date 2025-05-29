"""
Supabase database configuration and vector store setup
"""

import json
import logging
import os
from typing import Dict, List, Optional

from langchain.schema import Document
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
        self.table_name = "documents"

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        try:
            ids = []
            for doc in documents:
                # Create embedding for the document
                embedding = self.embeddings.embed_query(doc.page_content)
                
                # Prepare document data
                doc_data = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding
                }
                
                # Insert into database
                result = self.client.table(self.table_name).insert(doc_data).execute()
                
                if result.data:
                    ids.append(str(result.data[0]['id']))
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict] = None
    ) -> List[Document]:
        """Search for similar documents"""
        try:
            # Create embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Prepare RPC call parameters
            params = {
                "query_embedding": query_embedding,
                "match_count": k
            }
            
            if filter:
                params["filter"] = filter
            
            # Call the match_documents function
            response = self.client.rpc("match_documents", params).execute()
            
            # Convert results to Document objects
            documents = []
            for item in response.data:
                doc = Document(
                    page_content=item["content"],
                    metadata=item.get("metadata", {})
                )
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} similar documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores"""
        try:
            # Create embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Call the match_documents function
            params = {
                "query_embedding": query_embedding,
                "match_count": k
            }
            
            response = self.client.rpc("match_documents", params).execute()
            
            # Convert results to tuples of (Document, score)
            results = []
            for item in response.data:
                doc = Document(
                    page_content=item["content"],
                    metadata=item.get("metadata", {})
                )
                score = item.get("similarity", 0.0)
                results.append((doc, score))
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            raise
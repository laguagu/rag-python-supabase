"""
Embedding utilities for document processing and vector creation
"""

from dotenv import load_dotenv
import logging
import os
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
# Ensure environment variables are loaded
load_dotenv()


class EmbeddingManager:
    """Manages text embeddings and document processing"""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

        # Only initialize embeddings if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.embeddings = OpenAIEmbeddings(model=model_name)
        else:
            self.embeddings = None

        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def count_tokens(self, text: str) -> int:
        """Count approximate tokens in text using character count / 4"""
        return len(text) // 4  # Rough approximation: 1 token ≈ 4 characters

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks and return as strings"""
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise

    def split_text_into_chunks(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Split text into chunks suitable for embedding"""
        if metadata is None:
            metadata = {}

        try:
            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)

            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "token_count": self.count_tokens(chunk),
                    }
                )

                documents.append(Document(page_content=chunk, metadata=doc_metadata))

            logger.info(f"Split text into {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Failed to split text: {e}")
            raise

    def process_file(
        self, file_path: str, additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a file and return document chunks"""
        if additional_metadata is None:
            additional_metadata = {}

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Prepare metadata
            metadata = {
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": os.path.splitext(file_path)[1],
            }
            metadata.update(additional_metadata)

            # Split into chunks
            documents = self.split_text_into_chunks(content, metadata)

            logger.info(f"Processed file {file_path} into {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise

    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple files and return all document chunks"""
        all_documents = []

        for file_path in file_paths:
            try:
                documents = self.process_file(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue

        logger.info(
            f"Processed {len(file_paths)} files into {len(all_documents)} total chunks"
        )
        return all_documents

    def embed_query(self, query: str) -> List[float]:
        """Create embedding for a query"""
        if not self.embeddings:
            raise ValueError("OpenAI API key not available. Cannot create embeddings.")
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple documents"""
        if not self.embeddings:
            raise ValueError("OpenAI API key not available. Cannot create embeddings.")
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Created embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise

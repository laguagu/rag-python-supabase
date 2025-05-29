"""
RAG System - Simple MVP version
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from ..database.supabase_manager import SupabaseManager
    from ..embeddings.embedding_manager import EmbeddingManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from database.supabase_manager import SupabaseManager
    from embeddings.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class RAGSystem:
    """Simple RAG system for MVP"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.supabase_manager = SupabaseManager()
        self.embedding_manager = EmbeddingManager()

    def ask(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Ask a question and get an answer using the RAG system"""
        try:
            # 1. Retrieve relevant documents
            retrieved_docs = self.supabase_manager.similarity_search(
                query=query,
                k=4,  # Retrieve top 4 most relevant documents
            )

            # 2. Create context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"Document {i}:\n{doc.page_content}\n")

            context = (
                "\n".join(context_parts)
                if context_parts
                else "No relevant documents found."
            )

            # 3. Generate answer
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Olet avulias assistentti, joka vastaa kysymyksiin annetun kontekstin perusteella. 
            
Käytä seuraavaa kontekstia vastataksesi käyttäjän kysymykseen:

{context}

Jos et löydä vastausta kontekstista, sano että et löydä riittävästi tietoa vastaukseen.
Vastaa aina suomeksi, ellei käyttäjä pyydä muuta kieltä.""",
                    ),
                    ("human", "{query}"),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            answer = response.content

            logger.info(f"Generated answer for query: {query}")

            return {
                "query": query,
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "context": context,
            }

        except Exception as e:
            logger.error(f"RAG system failed: {e}")
            return {
                "query": query,
                "answer": "Anteeksi, tapahtui virhe kysymystä käsiteltäessä.",
                "retrieved_docs": [],
                "context": "",
            }

    def add_documents_from_files(self, file_paths: List[str]) -> bool:
        """Add documents from files to the knowledge base"""
        try:
            # Process files into document chunks
            documents = self.embedding_manager.process_multiple_files(file_paths)

            if not documents:
                logger.warning("No documents were processed")
                return False

            # Add to vector store
            self.supabase_manager.add_documents(documents)

            logger.info(
                f"Successfully added {len(documents)} document chunks from {len(file_paths)} files"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def add_text_document(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a text document to the knowledge base"""
        try:
            # Process text into chunks
            documents = self.embedding_manager.split_text_into_chunks(text, metadata)

            # Add to vector store
            self.supabase_manager.add_documents(documents)

            logger.info(
                f"Successfully added text document with {len(documents)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add text document: {e}")
            return False

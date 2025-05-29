"""
RAG System using LangGraph for orchestrating retrieval and generation
"""

import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, Document, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

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


# Define the state structure for our graph
class RAGState(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    retrieved_docs: List[Document]
    context: str
    answer: str


class RAGSystem:
    """Complete RAG system using LangGraph for workflow orchestration"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.supabase_manager = SupabaseManager()
        self.embedding_manager = EmbeddingManager()

        # Initialize the LangGraph workflow
        self.graph = None
        self._build_graph()

        # Initialize vector store
        self.supabase_manager.initialize_vector_store()

    def _build_graph(self):
        """Build the LangGraph workflow for RAG"""

        # Create the graph
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile the graph with memory
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    def _retrieve_documents(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant documents from the vector store"""
        query = state["query"]

        try:
            # Search for relevant documents
            retrieved_docs = self.supabase_manager.similarity_search(
                query=query,
                k=4,  # Retrieve top 4 most relevant documents
            )

            # Create context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"Document {i}:\n{doc.page_content}\n")

            context = "\n".join(context_parts)

            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")

            return {"retrieved_docs": retrieved_docs, "context": context}

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return {"retrieved_docs": [], "context": "No relevant documents found."}

    def _generate_answer(self, state: RAGState) -> Dict[str, Any]:
        """Generate an answer based on retrieved context"""
        query = state["query"]
        context = state["context"]

        # Create the prompt template
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

        try:
            # Generate the answer
            chain = prompt | self.llm
            response = chain.invoke({"context": context, "query": query})

            answer = response.content

            # Add to messages
            messages = [HumanMessage(content=query), AIMessage(content=answer)]

            logger.info(f"Generated answer for query: {query}")

            return {"answer": answer, "messages": messages}

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            error_answer = "Anteeksi, tapahtui virhe vastausta generoitaessa."
            return {
                "answer": error_answer,
                "messages": [
                    HumanMessage(content=query),
                    AIMessage(content=error_answer),
                ],
            }

    def ask(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Ask a question and get an answer using the RAG system"""

        # Prepare initial state
        initial_state = {
            "query": query,
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
        }
        # Configure thread for conversation memory
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Run the graph
            if self.graph is None:
                raise ValueError("RAG graph not initialized")

            result = self.graph.invoke(initial_state, config)  # type: ignore

            return {
                "query": query,
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"],
                "context": result["context"],
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

    def get_conversation_history(
        self, thread_id: str = "default"
    ) -> List[Dict[str, str]]:
        """Get conversation history for a thread"""
        # This would require implementing conversation storage
        # For now, return empty list
        return []

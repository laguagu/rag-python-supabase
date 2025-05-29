#!/usr/bin/env python3
"""
Quick test script to verify the RAG system is working
"""

import logging
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_system():
    """Run a quick test of the RAG system"""

    print("🧪 RAG System Quick Test\n")

    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False

    print("✅ Environment variables OK\n")

    try:
        # Import and initialize RAG system
        print("🔧 Initializing RAG system...")
        from src.rag.rag_system import RAGSystem

        rag_system = RAGSystem()
        print("✅ RAG system initialized\n")

        # Add test data
        print("📝 Adding test document...")
        test_text = """
        Tämä on testidokumentti RAG-järjestelmälle.
        Tämä dokumentti sisältää tietoa testauksesta.
        MVP (Minimum Viable Product) tarkoittaa minimaalista toimivaa tuotetta.
        RAG (Retrieval-Augmented Generation) yhdistää tiedonhaun ja tekstin generoinnin.
        """

        success = rag_system.add_text_document(
            test_text, metadata={"source": "test", "type": "test_document"}
        )

        if success:
            print("✅ Test document added successfully\n")
        else:
            print("❌ Failed to add test document\n")
            return False

        # Test retrieval
        print("🔍 Testing retrieval and generation...")
        test_query = "Mitä tarkoittaa MVP?"

        result = rag_system.ask(test_query)

        print(f"❓ Query: {test_query}")
        print(f"📚 Retrieved documents: {len(result['retrieved_docs'])}")
        print(f"🤖 Answer: {result['answer']}\n")

        # Test another query
        test_query2 = "Mikä on RAG?"
        result2 = rag_system.ask(test_query2)

        print(f"❓ Query: {test_query2}")
        print(f"📚 Retrieved documents: {len(result2['retrieved_docs'])}")
        print(f"🤖 Answer: {result2['answer']}\n")

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    success = test_rag_system()
    print("=" * 60)

    if success:
        print("\n🎉 RAG system is working correctly!")
        print("\n📚 Next steps:")
        print("1. Add more documents: python document_loader.py --file <file.txt>")
        print("2. Use CLI: python main.py")
        print("3. Use Web UI: streamlit run streamlit_app.py")
    else:
        print("\n❌ RAG system test failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Check your .env file has correct values")
        print("2. Run the SQL setup in Supabase dashboard")
        print("3. Check the error messages above")

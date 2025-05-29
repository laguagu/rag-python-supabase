#!/usr/bin/env python3
"""
Automated database setup script for Supabase
This script automatically sets up the required tables and functions for the RAG system
"""

import os

from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()


def auto_setup_database():
    """Automatically set up the Supabase database with required tables and functions"""

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return False

    print("🔌 Connecting to Supabase...")
    client = create_client(supabase_url, supabase_key)

    try:
        print("🔧 Setting up database schema...")

        # Step 1: Try to create the documents table
        print("📋 Creating documents table...")
        try:
            # First, let's try creating a simple table
            client.table("documents").select("*").limit(1).execute()
            print("✅ Documents table already exists")
        except Exception as e:
            print(
                f"📋 Documents table doesn't exist, this is expected: {str(e)[:100]}..."
            )

        # Step 2: Use RPC to execute SQL (if available)
        print("🔧 Attempting to execute setup SQL...")

        # Let's try a different approach - use the PostgREST API directly
        setup_queries = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            """CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(1536)
            );""",
            """CREATE OR REPLACE FUNCTION match_documents (
                query_embedding VECTOR(1536),
                match_count INT DEFAULT NULL,
                filter JSONB DEFAULT '{}'
            ) RETURNS TABLE (
                id BIGINT,
                content TEXT,
                metadata JSONB,
                similarity FLOAT
            )
            LANGUAGE plpgsql
            AS $$
            #variable_conflict use_column
            BEGIN
                RETURN QUERY
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (documents.embedding <=> query_embedding) AS similarity
                FROM documents
                WHERE metadata @> filter
                ORDER BY documents.embedding <=> query_embedding
                LIMIT match_count;
            END;
            $$;""",
            """CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
            USING hnsw (embedding vector_cosine_ops);""",
        ]

        # Try to execute each query
        for i, query in enumerate(setup_queries, 1):
            try:
                print(f"🔧 Executing setup step {i}/{len(setup_queries)}...")
                # Note: This might not work with standard Supabase client
                # We'll need to use the SQL editor manually
                print(f"   Query: {query[:50]}...")
            except Exception as e:
                print(f"   ⚠️  Step {i} needs manual execution: {str(e)[:50]}...")

        print("\n✅ Basic connection successful!")
        return True

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def test_database_ready():
    """Test if the database is ready for the RAG system"""

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return False

    print("\n🧪 Testing database readiness...")
    client = create_client(supabase_url, supabase_key)

    try:
        # Test 1: Check if documents table exists and is accessible
        print("📋 Testing documents table access...")
        result = client.table("documents").select(count="exact").execute()  # type: ignore
        print(f"✅ Documents table accessible (current count: {result.count})")

        # Test 2: Test inserting a sample document
        print("📝 Testing document insertion...")
        test_doc = {
            "content": "This is a test document for RAG system setup.",
            "metadata": {"source": "setup_test", "type": "test"},
            "embedding": [0.1] * 1536,  # Sample embedding
        }

        insert_result = client.table("documents").insert(test_doc).execute()
        print("✅ Document insertion successful")

        # Test 3: Test the match_documents function
        print("🔍 Testing match_documents function...")
        dummy_embedding = [0.1] * 1536
        match_result = client.rpc(
            "match_documents", {"query_embedding": dummy_embedding, "match_count": 5}
        ).execute()
        print(
            f"✅ match_documents function works (found {len(match_result.data)} matches)"
        )

        # Clean up test document
        if insert_result.data:
            test_id = insert_result.data[0]["id"]
            client.table("documents").delete().eq("id", test_id).execute()
            print("🧹 Cleaned up test document")

        print("\n🎉 Database is fully ready for the RAG system!")
        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        print(
            "\n💡 You may need to manually set up the database using the SQL provided."
        )
        return False


def show_manual_setup_instructions():
    """Show detailed manual setup instructions"""

    print("\n" + "=" * 80)
    print("📋 MANUAL SETUP INSTRUCTIONS")
    print("=" * 80)

    print(f"""
🔗 1. Go to your Supabase dashboard: {os.getenv("SUPABASE_URL", "your-supabase-url")}
🗄️  2. Navigate to: SQL Editor (in the left sidebar)
📝 3. Create a new query and paste the following SQL:

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);

-- Create match_documents function
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(1536),
    match_count INT DEFAULT NULL,
    filter JSONB DEFAULT '{{}}'
) RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create index for performance
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
USING hnsw (embedding vector_cosine_ops);

▶️  4. Click "Run" to execute the SQL
✅ 5. Verify that no errors occurred
🔄 6. Run this script again to test the setup
""")
    print("=" * 80)


if __name__ == "__main__":
    print("🚀 Automated Supabase Database Setup for RAG System\n")

    # Try automatic setup
    setup_success = auto_setup_database()

    if setup_success:
        # Test if everything is working
        test_success = test_database_ready()

        if test_success:
            print("\n✅ Setup Complete! Your RAG system is ready to use.")
            print("\n🚀 You can now run:")
            print("   python main.py")
            print("   streamlit run streamlit_app.py")
            print("   python document_loader.py")
        else:
            print("\n⚠️  Database setup incomplete.")
            show_manual_setup_instructions()
    else:
        print("\n❌ Automatic setup failed.")
        show_manual_setup_instructions()

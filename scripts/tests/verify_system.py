#!/usr/bin/env python3
"""
Quick verification script to check if the RAG system is working
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def verify_environment():
    """Verify that all required environment variables are set"""
    print("🔍 Checking environment variables...")

    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(10, len(value))}...")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)

    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        return False

    print("✅ All environment variables are set")
    return True


def verify_imports():
    """Verify that all required packages can be imported"""
    print("\n🔍 Checking package imports...")

    try:
        import openai

        print("✅ openai")
    except ImportError as e:
        print(f"❌ openai: {e}")
        return False

    try:
        from supabase import create_client

        print("✅ supabase")
    except ImportError as e:
        print(f"❌ supabase: {e}")
        return False

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        print("✅ langchain")
    except ImportError as e:
        print(f"❌ langchain: {e}")
        return False

    try:
        from langchain_openai import OpenAIEmbeddings

        print("✅ langchain-openai")
    except ImportError as e:
        print(f"❌ langchain-openai: {e}")
        return False

    try:
        import streamlit

        print("✅ streamlit")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
        return False

    print("✅ All packages imported successfully")
    return True


def verify_openai_connection():
    """Test OpenAI API connection"""
    print("\n🔍 Testing OpenAI connection...")

    try:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=SecretStr(api_key) if api_key else None,
        )

        # Test embedding generation
        test_text = "This is a test for OpenAI embeddings"
        embedding = embeddings.embed_query(test_text)

        print(
            f"✅ OpenAI connection successful (embedding dimension: {len(embedding)})"
        )
        return True

    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False


def verify_supabase_connection():
    """Test Supabase connection and database setup"""
    print("\n🔍 Testing Supabase connection...")

    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

        client = create_client(supabase_url, supabase_key)

        # Test basic connection
        print("📡 Testing basic connection...")

        # Test documents table
        print("📋 Testing documents table...")
        result = client.table("documents").select("count", count="exact").execute() # type: ignore
        print(f"✅ Documents table exists (count: {result.count})")

        # Test match_documents function
        print("🔍 Testing match_documents function...")
        dummy_embedding = [0.0] * 1536
        result = client.rpc(
            "match_documents", {"query_embedding": dummy_embedding, "match_count": 1}
        ).execute()
        print("✅ match_documents function works")

        print("✅ Supabase database is properly configured")
        return True

    except Exception as e:
        print(f"❌ Supabase test failed: {e}")
        print(
            "\n💡 You may need to run the database setup SQL manually in Supabase dashboard"
        )
        return False


def run_full_verification():
    """Run all verification checks"""
    print("🚀 RAG System Verification\n")
    print("=" * 60)

    checks = [
        ("Environment Variables", verify_environment),
        ("Package Imports", verify_imports),
        ("OpenAI Connection", verify_openai_connection),
        ("Supabase Database", verify_supabase_connection),
    ]

    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))

    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ Your RAG system is ready to use!")
        print("\n🚀 Available commands:")
        print("   python main.py                    # CLI interface")
        print("   streamlit run streamlit_app.py    # Web interface")
        print("   python document_loader.py         # Load sample documents")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\n📋 Next steps:")
        print("1. Fix any failed checks above")
        print("2. For Supabase issues, run the SQL setup manually")
        print("3. Re-run this verification script")

    return all_passed


if __name__ == "__main__":
    run_full_verification()

-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store your documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT, -- corresponds to Document.page_content
    metadata JSONB, -- corresponds to Document.metadata
    embedding VECTOR(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Drop existing function if it exists (to avoid return type conflicts)
DROP FUNCTION IF EXISTS match_documents(vector, integer, jsonb);

-- Create a function to search for documents
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(1536),
    match_count INT DEFAULT NULL,
    filter JSONB DEFAULT '{}'
) RETURNS TABLE (
    id UUID,
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
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE documents.metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create an index for faster similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
USING hnsw (embedding vector_cosine_ops);
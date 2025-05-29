-- 1. Ota pgvector käyttöön
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Poista vanha funktio JOS SE ON OLEMASSA
DROP FUNCTION IF EXISTS match_documents(vector, integer, jsonb);

-- 3. Poista vanha taulu jos on olemassa
DROP TABLE IF EXISTS documents CASCADE;

-- 4. Luo documents taulu
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);

-- 5. Luo match_documents funktio
CREATE OR REPLACE FUNCTION match_documents (
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
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE metadata @> filter
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 6. Luo indeksi
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
USING hnsw (embedding vector_cosine_ops);
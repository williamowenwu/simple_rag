CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content text,
    embedding vector(768),
    metadata jsonb
)

CREATE TABLE session(
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    history JSONB NOT NULL DEFAULT '[]',
)
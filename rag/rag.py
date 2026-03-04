import asyncio
from contextlib import asynccontextmanager
from collections import defaultdict
import logging
from uuid import UUID

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi, BM25
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db import get_conn

# vector database
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
# Needs more than 4gb vram
QWEN = 'hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:latest'
client = ollama.AsyncClient(host="http://host.docker.internal:11434")

# what do i need to be an api? there are somethings that don't necessarily need to be an api
# anything that user interacts with the model (uploading things) :
# 1. prompts/text
# 1a. images (not yet)
# i think its just the streaming of prompts back and forth. we don't need the underlying RAG logic to be exposed i think

bm25: BM25 | None = None
chunks: list[str] | None = None

class UserPrompt(BaseModel):
    prompt: str

@asynccontextmanager
async def lifespan(app: FastAPI): 
    global chunks
    # can call await directly becauses its async
    chunks = await chunk_data('cat.txt')
    # await add_dataset_to_db(chunks)
    await add_dataset_to_db_hybrid(chunks)
    yield

app = FastAPI(lifespan=lifespan)

@app.post('/chat', response_class=StreamingResponse)
async def start_chat(query: UserPrompt):
    # start a new session
    initial_history = [{'role': 'user','content': query.prompt}]
    async with get_conn() as conn:
        async with conn.cursor() as curr:
            cursor = await curr.execute(
                """
                INSERT INTO session (history)
                VALUES (%s)
                RETURNING session_id;
                """,
                (Jsonb(initial_history),)
            )
            row = await cursor.fetchone()
            session_id = row[0]
    
    return StreamingResponse(prompt_llm(session_id, initial_history), media_type='text/plain')
    

@app.post('/chat/{session_id}', response_class=StreamingResponse)
async def continue_chat(session_id: UUID, query: UserPrompt):
    # obtain history from the database
    new_query = {'role': 'user', 'content': query.prompt}
    async with get_conn() as conn:
        async with conn.cursor() as curr:
            history = await curr.execute(
                """
                UPDATE session
                SET history = history || %s::jsonb
                WHERE session_id = %s
                RETURNING history;
                """
            , (Jsonb(new_query), session_id)
            )

            row = await history.fetchone() # Returns a tuple in order of what you selected
            history = row[0]

            return StreamingResponse(prompt_llm(session_id, history), media_type='text/plain') 

async def prompt_llm(session_id, history: list[dict[str:str]]):
    knowledge = await hybrid_retrieve(history[-1]['content'])
    print('Retrieved Knowledge:')
    for chunk, similarity in knowledge:
        print(f' - (rrf: {similarity:.4f}) {chunk}')
        # print(f' - (similarity: {similarity:.4f}) {chunk}')
    res = {}
    res['knowledge'] = knowledge
    res['res'] = []

    instruction_prompt = f""" 
    You are a fact retrieval assistant. You MUST answer ONLY using the facts listed below. 
    If the answer is not in the list, say "I don't know."
    Do NOT use any outside knowledge. Do NOT make things up.
    {'\n'.join([f' - {chunk}' for chunk, _ in knowledge])}
    """
    try:
 
        stream = await asyncio.wait_for(
            client.chat(
                # model=LANGUAGE_MODEL,
                model=QWEN,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    *history
                ],
                stream=True
                ),
            timeout=10)

        collected = []
        async for chunk in stream:
            content = chunk['message']['content']
            collected.append(content)
            yield content
         
        response = {'role': 'assistant', 'content' :"".join(collected)}
        # then add collected to the prompt and save
        
        # save back response
        async with get_conn() as conn:
            async with conn.cursor() as curr:
                await curr.execute(
                    """
                    UPDATE session
                    -- append too the jsonb 'content' array
                    SET history = history || %s::jsonb
                    WHERE session_id = %s
                    """,
                    (Jsonb(response), session_id)
                )
        print(f"Session id for last chat: {session_id}")

    except TimeoutError as te:
        logging.error(f'error: {te}')
        yield "Error Request Timeout"
    except Exception as e:
        logging.error(f'base error: {e}')
        raise
    # res['status'] = 'success'
    # return res


async def chunk_data(filepath: str) -> list[str]:
    """
    Uses recursive chunking strategy with sentence boundaries. Keep it simple
    """
    with open(filepath) as f:
        splitter = RecursiveCharacterTextSplitter(['.', ',',r'\n'], chunk_size=400, chunk_overlap=60)
        #todo: im worried about doing readlines it will just load everything into memory idk if it scales
        chunked_data = splitter.split_text(f.read())
        print(f"Loading {len(chunked_data)} entries...")
        return chunked_data

async def add_dataset_to_db_hybrid(data_set: list[str]) -> None:
    async with get_conn() as conn:
        global bm25
        async with conn.cursor() as cur:
            await cur.execute(
                """
                TRUNCATE TABLE chunks;
                TRUNCATE TABLE session;              
                """
            )

            bm25 = BM25Okapi([sentence.split() for sentence in data_set])

            for i, chunk in enumerate(data_set): 
                embeddings = await client.embed(EMBEDDING_MODEL, input=chunk)
                embedding = embeddings['embeddings'][0]
                
                await cur.execute(
                    """
                    INSERT INTO chunks
                    (content, embedding, metadata) 
                    VALUES (%s, %s, %s) 
                    """,
                    (chunk, embedding, Jsonb({}))
                ) 
                print(f"Added chunk {i+1}/{len(data_set)} to vector db")

async def rrf(semantic_res: list[tuple[float,str]], bm25_res: list[tuple[float]], k=60) -> tuple[float,str]:
    """
    Reciprocal rank fusion
    """
    rrf_scores = defaultdict(float)
    #* ah i see, the rank is really just the index of each list sorted
 
    for rank, semantic_chunk in enumerate(semantic_res, start=1):
        score_contribution = 1 / (k + rank)
        bm25_chunk = bm25_res[rank-1][1]
        rrf_scores[semantic_chunk[1]] += score_contribution
        rrf_scores[bm25_chunk] += score_contribution
    
    # return the dict sorted by the values
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


async def hybrid_retrieve(query: str, top:int=3) -> list[tuple]:
    query_embedding = await client.embed(EMBEDDING_MODEL, input=query)
    query_embeded = query_embedding['embeddings'][0]
    semantic_res = None

    async with get_conn() as conn: 
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1 - (embedding <=> %s::vector) as cosine_similarity, content
                FROM chunks
                ORDER BY cosine_similarity DESC
                LIMIT %s
                """,
                (query_embeded, top)
            )
    
            # no need to convert?
            semantic_res = await cur.fetchall()

    tokenized_query = query.split()
    doc_scores_index = bm25.get_scores(tokenized_query)
    
    # map the index with the chunk together
    doc_scores = [(doc_scores_index[i], chunks[i]) for i in range(len(chunks))]
    doc_scores = sorted(doc_scores, key=lambda x: x[0], reverse=True)[:top]
    return await rrf(semantic_res, doc_scores)

# Limitations from what i learned
"""
The model actually ignores or doesn't follow instruction commands very well. it either ignores it completely or ocassionally gives a context answer
This is dependent on the user query.

This limitation comes from the model parameter size of 1B. Tiny models are really notorious for not following instruction prompts. 
"""
"""
cosine similarity measures the similarity between two non zero vectors by calculating the cosine angle between them.
The range is -1 to 1. 1 is identical. -1 is exactly opposite 
helps find document similarity regardless of length
semantic search for vector embeddings closest to user's query
"""

# SYNCHRONOUS/OLD FOR COMPARISON

# line by line chunking strategy
async def load_data(filepath: str) -> list[str]:
    with open(filepath) as f:
        dataset = f.readlines()
        print(f"Loading {len(dataset)} entries...")
        return dataset

async def add_dataset_to_db(data_set: list[str]) -> None:
    # Regular embedding without rrf
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE TABLE chunks")
            # chunking strategy of one line
            for i, chunk in enumerate(data_set): 
                embeddings = await client.embed(EMBEDDING_MODEL, input=chunk)
                embedding = embeddings['embeddings'][0]
                
                await cur.execute(
                    """
                    INSERT INTO chunks
                    (content, embedding, metadata) 
                    VALUES (%s, %s, %s) 
                    """,
                    (chunk, embedding, Jsonb({}))
                ) 
                print(f"Added chunk {i+1}/{len(data_set)} to vector db")

# obtains top x similarities in search and returns them
async def retrieve(query: str, top:int=3) -> list[tuple]:
    query_embedding = await client.embed(EMBEDDING_MODEL, input=query)
    query_embeded = query_embedding['embeddings'][0]

    async with get_conn() as conn: 
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT content, 1 - (embedding <=> %s::vector) as cosine_similarity
                FROM chunks
                ORDER BY cosine_similarity DESC
                LIMIT %s
                """,
                (query_embeded, top)
            )
            # no need to convert?
            return await cur.fetchall()
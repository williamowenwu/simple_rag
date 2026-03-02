import asyncio
from contextlib import asynccontextmanager
from collections import defaultdict
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg.types.json import Jsonb
from rank_bm25 import BM25Okapi, BM25
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db import get_conn

# vector database
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
QWEN = 'hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:latest'

bm25: BM25 | None = None
chunks: list[str] | None = None

class UserPrompt(BaseModel):
    prompt: str

@asynccontextmanager
async def lifespan(app: FastAPI): 
    global chunks # can call await directly becauses its async
    chunks = await chunk_data('cat.txt')
    # await add_dataset_to_db(chunks)
    await add_dataset_to_db_hybrid(chunks)
    yield

app = FastAPI(lifespan=lifespan)

# what do i need to be an api? there are somethings that don't necessarily need to be an api
# anything that user interacts with the model (uploading things) :
# 1. prompts/text
# 1a. images (not yet)
# i think its just the streaming of prompts back and forth. we don't need the underlying RAG logic to be exposed i think

client = ollama.AsyncClient(host="http://host.docker.internal:11434")
# line by line chunking strategy
async def load_data(filepath: str) -> list[str]:
    with open(filepath) as f:
        dataset = f.readlines()
        print(f"Loading {len(dataset)} entries...")
        return dataset

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

async def add_dataset_to_db(data_set: list[str]) -> None:
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

async def add_dataset_to_db_hybrid(data_set: list[str]) -> None:
    async with get_conn() as conn:
        global bm25
        async with conn.cursor() as cur:
            await cur.execute("TRUNCATE TABLE chunks")

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

@app.post('/chat', response_class=StreamingResponse)
async def start_chat(prompt: UserPrompt):
    
    knowledge = await hybrid_retrieve(prompt.prompt)
    print('Retrieved Knowledge:')

    res = {}
    res['knowledge'] = knowledge
    res['res'] = []

    instruction_prompt = f""" 
    You are a fact retrieval assistant. You MUST answer ONLY using the facts listed below. 
    If the answer is not in the list, say "I don't know."
    Do NOT use any outside knowledge. Do NOT make things up.
    {'\n'.join([f' - {chunk}' for chunk, _ in knowledge])}
    """
 
    stream = await client.chat(
        # model=LANGUAGE_MODEL,
        model=QWEN,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': prompt.prompt},
        ],
        stream=True
    ) 

    async for chunk in stream:
        yield chunk['message']['content']
    # res['status'] = 'success'
    # return res

# Generation Phase
async def prompt_user():
    user_query = input("\n\nAsk me a question about cats dumbass: ")
    retrieved_knowledge = await retrieve(user_query)
    # retrieved_knowledge = await hybrid_retrieve(user_query)

    print('Retrieved Knowledge: ')
    for chunk, similarity in retrieved_knowledge:
        # print(f' - (rrf: {similarity:.4f}) {chunk}')
        print(f' - (similarity: {similarity:.4f}) {chunk}')
    
    instruction_prompt = f'''
     You are a fact retrieval assistant. You MUST answer ONLY using the facts listed below. 
     If the answer is not in the list, say "I don't know."
    Do NOT use any outside knowledge. Do NOT make things up.
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''
    # print(f'\n{instruction_prompt}')

    stream = await client.chat(
        # model=LANGUAGE_MODEL,
        model=QWEN,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': user_query},
        ],
        stream=True
    ) 

    print("Chatbot's response: ")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

async def main():
    global chunks
    chunks = await chunk_data('cat.txt')
    await add_dataset_to_db(chunks)
    # await add_dataset_to_db_hybrid(chunks)
    
    while True:
        await prompt_user()

if __name__ == '__main__':
    # chunks = load_data('cat.txt')
    asyncio.run(main())


# Limitations from what i learned
"""
The model actually ignores or doesn't follow instruction commands very well. it either ignores it completely or ocassionally gives a context answer
This is dependent on the user query.

This limitation comes from the model parameter size of 1B. Tiny models are really notorious for not following instruction prompts. 
"""

def cosine_similarity(a,b):
    """
    cosine similarity measures the similarity between two non zero vectors by calculating the cosine angle between them.
    The range is -1 to 1. 1 is identical. -1 is exactly opposite 
    helps find document similarity regardless of length
    semantic search for vector embeddings closest to user's query
    """
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


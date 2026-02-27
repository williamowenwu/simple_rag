import ollama
import psycopg
from pgvector import register_vector


# vector database
VECTOR_DB = [] # we're just using a list?
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

client = ollama.Client(host="http://host.docker.internal:11434")
# load data set
def load_data(filepath: str) -> list[str]:
    with open(filepath) as f:
        dataset = f.readlines()
        print(f"Loading {len(dataset)} entries...")
        return dataset


def add_chunk_to_db(chunk: str) -> None:
    embed = client.embed(EMBEDDING_MODEL, input=chunk)
    embeddings = embed['embeddings'][0]

    with psycopg.connect("dbname=postgres user=postgres") as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            
            cur.execute(
                """
                INSERT INTO chunks
                (content, embedding, metadata) 
                VALUES (%s, %s, %s), 
                """,
                (chunk, embeddings, {})
            )

    # VECTOR_DB.append((chunk, embeddings))

def add_dataset_to_db(data_set: list[str]) -> None:
    for i, chunk in enumerate(data_set): # using a line for simplicity?
        add_chunk_to_db(chunk) 
        print(f"Added chunk {i+1}/{len(data_set)} to vector db")

"""
cosine similarity measures the similarity between two non zero vectors by calculating the cosine angle between them.
The range is -1 to 1. 1 is identical. -1 is exactly opposite 
helps find document similarity regardless of length
semantic search for vector embeddings closest to user's query
"""
def cosine_similarity(a,b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# obtains top x similarities in search and returns them
def retrieve(query, top=3) -> list[tuple]:
    query_embedding = client.embed(EMBEDDING_MODEL, input=query)
    query_embeded = query_embedding['embeddings'][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(embedding, query_embeded)
        similarities.append((chunk, similarity))
 
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top]


# Generation Phase
def prompt_user():
    user_query = input("\n\nAsk me a question about cats dumbass: ")
    retrieved_knowledge = retrieve(user_query)

    print('Retrieved Knowledge: ')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.4f}) {chunk}')
    
    instruction_prompt = f'''
    # You are a fact retrieval assistant. You MUST answer ONLY using the facts listed below. 
    # If the answer is not in the list, say "I don't know."
    # Do NOT use any outside knowledge. Do NOT make things up.
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''
    # print(f'\n{instruction_prompt}')

    stream = client.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': user_query},
        ],
        stream=True
    ) 

    print("Chat bot response: ")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == '__main__':
    chunks = load_data('cat.txt')
    add_dataset_to_db(chunks)
    
    while True:
        prompt_user()


# Limitations from what i learned
"""
The model actually ignores or doesn't follow instruction commands very well. it either ignores it completely or ocassionally gives a context answer
This is dependent on the user query.

This limitation comes from the model parameter size of 1B. Tiny models are really notorious for not following instruction prompts. 
"""


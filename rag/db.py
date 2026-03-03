import os
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from psycopg import AsyncConnection
from pgvector.psycopg import register_vector_async


DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", 'postgres')
DB_USER = os.getenv("POSTGRES_USER",'postgres')

@asynccontextmanager
async def get_conn() -> AsyncIterator[AsyncConnection]:
    conn = None
    try:
        # what happens if there is an error at this part?
        conn = await AsyncConnection.connect(f"host=postgres user={DB_USER} password={DB_PASSWORD} dbname=postgres", connect_timeout=1)
        await register_vector_async(conn)
        yield conn
    except BaseException as e:
        logging.error(f"Exception: {e}",)
        await conn.rollback()
        raise
    else:
        await conn.commit()
    finally:
        if conn:
            await conn.close()
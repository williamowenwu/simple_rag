import os
import logging
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import register_vector


DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", 'postgres')
DB_USER = os.getenv("POSTGRES_USER",'postgres')

@contextmanager
def get_conn():
    conn = None
    try:
        # what happens if there is an error at this part?
        conn = psycopg.connect(f"host=postgres user={DB_USER} password={DB_PASSWORD} dbname=postgres")
        register_vector(conn)
        yield conn
    # i probably shouldn't do this
    except Exception as e:
        logging.error(f"Exception: {e}",)
        raise
    finally:
        if conn:
            conn.close()
import psycopg2
import pandas as pd

from .db_config import DB_CONFIG  


def load_total_hiking_data(limit=None):
    query = "SELECT * FROM public.total_hiking_data"
    if limit:
        query += f" LIMIT {limit}"
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(query, conn)
    return df

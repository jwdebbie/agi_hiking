import psycopg2
import pandas as pd

from .db_config import DB_CONFIG  


def load_total_hiking_data(limit=None, filter_missing_address=False):
    query = "SELECT * FROM public.total_hiking_data"
    
    # address NULL 제외
    if filter_missing_address:
        query += " WHERE address IS NOT NULL"
    
    if limit:
        query += f" LIMIT {limit}"
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(query, conn)
    
    return df


def load_member_data(version='original', limit=None):
    table_map = {
        'original': 'member',
        'cleaned': 'member_cleaned',
        'imputed': 'member_imputed'
    }
    
    table_name = table_map.get(version, 'member')
    query = f"SELECT * FROM {table_name}"
    
    if limit:
        query += f" LIMIT {limit}"
    
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(query, conn)
    
    return df
import os
from dotenv import load_dotenv
import pymysql
import pandas as pd
from sandhya_aqua.utils.sql_queries import queries
from urllib.parse import urlparse

load_dotenv()


def get_db_connection():
    conn_str = os.getenv("MYSQL_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("MYSQL_CONNECTION_STRING not set in .env")

    if conn_str.startswith("mysql+pymysql://"):
        conn_str = conn_str.replace("mysql+pymysql://", "mysql://", 1)

    result = urlparse(conn_str)
    if result.scheme != "mysql":
        raise ValueError("Connection string must start with mysql://")

    return pymysql.connect(
        host=result.hostname,
        user=result.username,
        password=result.password,
        database=result.path.lstrip("/"),
        port=result.port or 3306,
        cursorclass=pymysql.cursors.DictCursor,
    )


def run_predefined_query(query_key, params=None):
    if query_key not in queries:
        raise ValueError(f"Query key '{query_key}' not found.")
    query = queries[query_key]
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            df = pd.DataFrame(rows)
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    df = run_predefined_query("packing_yield_query", params=("5P2-464",))
    df.to_json("new.json", index=False)
    print(df.head(n=5))

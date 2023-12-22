import pandas as pd
import chardet
import pymysql
import sqlalchemy
from sqlalchemy import create_engine

hot100 = pd.read_csv("파일명", encoding='latin-1')
pymysql.install_as_MySQLdb()

USER = "root"
PASSWORD = "pwd"
ENDPOINT = "host name"
SCHEMA = "dbname"


conn = pymysql.connect(
    host=ENDPOINT,
    user=USER,
    password=PASSWORD,
    port=3306,
    cursorclass=pymysql.cursors.DictCursor,
)

cur = conn.cursor()

cur.execute(f"CREATE DATABASE IF NOT EXISTS {SCHEMA};")

engine = create_engine(f"mysql+mysqldb://{USER}:{PASSWORD}@{ENDPOINT}:3306")

hot100.to_sql(
    "table name",
    engine,
    schema=SCHEMA,
    index=False,
    if_exists="replace",
    chunksize=10000,
)


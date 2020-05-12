import os
import sys
sys.path.append("..")
from configs import config
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

cfg = config.read()
host = cfg.get("postgres", "host")
database = cfg.get("postgres", "database")
user = cfg.get("postgres", "user")
password = cfg.get("postgres", "password")
table = cfg.get("postgres", "table")
pk = cfg.get('postgres', 'primary_key')
port = 5432


def _getpostgres_connection():
    """
    """
    conn_str = "host={} dbname={} user={} password={}".format(host, database, user, password)
    conn = psycopg2.connect(conn_str)

    return conn


def postgres_to_dataframe_with_limits(table=table, limit='ALL', offset=0):
    conn = _getpostgres_connection()

    sql_string = "select * from " + table + " order by " + pk + " limit " + str(
        limit) + " offset " + str(offset)
    data = pd.read_sql(sql_string, con=conn)
    conn.close()
    return data



def postgres_to_dataframe(table=table):
    """

    """
    conn = _getpostgres_connection()
    data = pd.read_sql('select * from ' + table, con=conn)
    conn.close()
    return data


def get_postgres_column(column, table=table):
    conn = _getpostgres_connection()
    data = pd.read_sql('select ' + column + ' from ' + table, con=conn)
    conn.close()
    return data


def delete_table(tablename):
    conn = _getpostgres_connection()
    cur = conn.cursor()

    delete = """Drop table if exists """ + tablename
    cur.execute(delete)
    conn.commit()
    conn.close()


def updated_input_dataframe_to_postgres(df, tablename):
    """
    
    Arguments:
    - `filename`:
    """
    # tablename = 'documents'

    conn = _getpostgres_connection()
    cur = conn.cursor()

    delete = """Drop table if exists """ + tablename
    cur.execute(delete)
    conn.commit()

    connection_string = "postgresql+psycopg2://" + user + ":" + password + "@" + \
                        host + ":" + str(port) + "/" + database

    engine = create_engine(connection_string)
    df.to_sql(tablename, con=engine, index=False)
    conn.close()


def dataframe_to_postgres(df, tablename, db_append):
    connection_string = "postgresql+psycopg2://" + user + ":" + password + "@" + \
                        host + ":" + str(port) + "/" + database
    engine = create_engine(connection_string)
    if db_append.lower() == 'true':
        df.to_sql(tablename, con=engine, if_exists='append', index=False)
    else:
        df.to_sql(tablename, con=engine, if_exists='replace', index=False)
    engine.dispose()


# create table in postgres
# 'CREATE TABLE "similarity"(index1 varchar(20) 
#  NOT NULL, index2 varchar(20) NOT NULL, sim varchar(20) NOT NULL)'

def csv_to_postgres(csv_path, id):
    if (id == 1):
        create_table_string = "Create Table similarity1(index1 varchar(100), index2 varchar(100), similarity varchar(100));"
        similarity_string = "COPY similarity1 FROM '" + csv_path + "' DELIMITERS ',';"
        delete = """Drop table if exists similarity1"""
    elif (id == 2):
        create_table_string = "Create Table similarity2(index1 varchar(100), index2 varchar(100), similarity varchar(100));"
        similarity_string = "COPY similarity2 FROM '" + csv_path + "' DELIMITERS ',';"
        delete = """Drop table if exists similarity2"""
    else:
        create_table_string = "Create Table similarity3(passages varchar(100), similarity_score varchar(100));"
        similarity_string = "COPY similarity3 FROM '" + csv_path + "' DELIMITERS ',';"
        delete = """Drop table if exists similarity3"""

    conn = _getpostgres_connection()
    cur = conn.cursor()

    cur.execute(delete)

    cur.execute(create_table_string)
    conn.commit()

    csv_path = os.path.abspath(csv_path)
    cur.execute(similarity_string)
    conn.commit()

    cur.close()
    conn.close()

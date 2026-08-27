import streamlit as st
import mysql.connector
from mysql.connector import Error

MYSQL_CONFIG = {
    "host": st.secrets["MYSQL_HOST"],
    "port": int(st.secrets["MYSQL_PORT"]),
    "user": st.secrets["MYSQL_USER"],
    "password": st.secrets["MYSQL_PASSWORD"]
}

def get_connection():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def init_db():
    conn = get_connection()

    if conn:
        cursor = conn.cursor()

        # Buat database jika belum ada
        cursor.execute("USE railway")

        # Buat tabel dataset
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                original_tweet TEXT,
                clean_tweet TEXT,
                label VARCHAR(20)
            )
        """)

        # Buat tabel hasil evaluasi
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_evaluation (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(50),
                k_fold INT,
                accuracy FLOAT,
                precision_score FLOAT,
                recall_score FLOAT,
                f1_score FLOAT
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()


def get_db_connection():
    return mysql.connector.connect(
        host="hayabusa.proxy.rlwy.net",
        port=43891,
        user="root",
        password="PASSWORD_MYSQL_RAILWAY",
        database="sentiment_indihome"
    )

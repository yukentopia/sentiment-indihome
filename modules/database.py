import streamlit as st
import mysql.connector


def get_db_connection():
    conn = mysql.connector.connect(
        host=st.secrets["MYSQL_HOST"],
        port=int(st.secrets["MYSQL_PORT"]),
        user=st.secrets["MYSQL_USER"],
        password=st.secrets["MYSQL_PASSWORD"],
        database="railway"
    )

    cursor = conn.cursor()

    # Tabel dataset
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_tweets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_tweet TEXT,
            clean_tweet TEXT,
            label VARCHAR(20),
            sentiment_score INT
        )
    """)

    # Tabel hasil evaluasi
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluation (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(50),
            k_fold INT,
            accuracy FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            use_smote BOOLEAN DEFAULT FALSE,
            f1_macro FLOAT,
            balanced_acc FLOAT,
            mcc FLOAT,
            confusion_matrix TEXT
        )
    """)

    # Tabel user login
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL,
            nama_lengkap VARCHAR(100) NOT NULL
        )
    """)

    # Tabel kamus khusus
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_lexicon (
            id INT AUTO_INCREMENT PRIMARY KEY,
            word VARCHAR(255) UNIQUE,
            score INT
        )
    """)

    conn.commit()
    cursor.close()

    return conn

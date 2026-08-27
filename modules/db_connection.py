import mysql.connector
from mysql.connector import Error

def get_connection():
    try:
        # Sesuaikan dengan kredensial MySQL kamu
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            # database='sentiment_indihome' # Jangan di-set dulu saat inisialisasi awal
        )
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def init_db():
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        # Buat database jika belum ada
        cursor.execute("CREATE DATABASE IF NOT EXISTS sentiment_indihome")
        cursor.execute("USE sentiment_indihome")
        
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
        host='localhost',
        user='root',
        password='',
        database='sentiment_indihome'
    )
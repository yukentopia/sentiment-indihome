import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",        # Sesuaikan dengan user XAMPP/MySQL kamu
        password="",        # Biasanya kosong jika pakai XAMPP default
        database="sentiment_indihome" # Ganti dengan nama database skripsimu
    )
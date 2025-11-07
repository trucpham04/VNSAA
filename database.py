import sqlite3
import streamlit as st
import pandas as pd

from constant import DB_NAME

# =========================== Database Initialization ===========================
def initialize_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

# =========================== Database Saving ===========================
def save_to_sqlite(data: dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sentiments (text, sentiment) VALUES (?, ?)",
            (data['text'], data['sentiment'])
        )
        conn.commit()
    except Exception as e:
        st.error(f"Error saving to SQLite: {e}")
    finally:
        if conn:
            conn.close()

# =========================== Database Loading ===========================
def load_data_from_sqlite() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM sentiments ORDER BY id DESC LIMIT 50", conn)
        return df
    except Exception as e:
        st.error(f"Error loading data from SQLite: {e}")
        return pd.DataFrame(columns=["id", "text", "sentiment", "timestamp"])
    finally:
        if conn:
            conn.close()

def delete_all_records():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sentiments")
        conn.commit()
    except Exception as e:
        st.error(f"Error deleting record from SQLite: {e}")
    finally:
        if conn:
            conn.close()
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
def load_data_from_sqlite(last_id: int = None, page_size: int = 50) -> pd.DataFrame:
    """
    Load data from SQLite with cursor-based pagination.
    
    Args:
        last_id: ID of the last record from previous page (None for first page)
        page_size: Number of records to return (default: 50)
    
    Returns:
        DataFrame with sentiment records
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        if last_id is None:
            # First page: get the most recent records
            query = "SELECT * FROM sentiments ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(page_size,))
        else:
            # Next page: get records with id < last_id
            query = "SELECT * FROM sentiments WHERE id < ? ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(last_id, page_size))
        return df
    except Exception as e:
        st.error(f"Error loading data from SQLite: {e}")
        return pd.DataFrame(columns=["id", "text", "sentiment", "timestamp"])
    finally:
        if conn:
            conn.close()

def has_more_records(last_id: int) -> bool:
    """
    Check if there are more records after the given last_id.
    
    Args:
        last_id: ID of the last record in current page
    
    Returns:
        True if there are more records, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentiments WHERE id < ?", (last_id,))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        st.error(f"Error checking for more records: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_total_pages(page_size: int = 50) -> int:
    """
    Get total number of pages based on total records and page size.
    
    Args:
        page_size: Number of records per page (default: 50)
    
    Returns:
        Total number of pages
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentiments")
        total_records = cursor.fetchone()[0]
        total_pages = (total_records + page_size - 1) // page_size  # Ceiling division
        return max(1, total_pages)  # At least 1 page even if empty
    except Exception as e:
        st.error(f"Error getting total pages: {e}")
        return 1
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
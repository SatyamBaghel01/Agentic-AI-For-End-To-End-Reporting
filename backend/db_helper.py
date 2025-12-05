import sqlite3
import os

# SQLite database path
DB_PATH = os.path.join(os.path.dirname(__file__), "steel.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access
    return conn

def execute_query(query, params=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        # If it's a SELECT query, return rows
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            return {"success": True, "data": data}

        # For INSERT/UPDATE/CREATE queries
        conn.commit()
        return {"success": True, "message": "Query executed successfully"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        conn.close()

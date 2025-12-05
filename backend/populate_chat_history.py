import sqlite3
from datetime import datetime, timedelta
import random
import json

DB_NAME = "steel.db"

users = [
    "operator01", "shift_supervisor", "energy_analyst",
    "process_engineer", "maintenance_team"
]

questions = [
    "Show power usage trend for last 7 days",
    "How many machines are running in Hot Rolling division?",
    "Which machine had highest spindle load today?",
    "Compare voltage fluctuation between plants",
    "Show average motor temperature",
    "List machines with frequent alarms",
    "Find total energy consumption per plant",
    "Show speed trend for CR_Mill_01"
]

explanations = [
    "Below is the detailed analysis from machine telemetry.",
    "Machine data processed and summarized as requested.",
    "Trend analysis based on sensor and energy readings.",
    "Computed using real-time logged operational data.",
    "Here is the result from the last processing interval."
]

graphs = [
    "Line chart showing variation over time",
    "Bar chart comparing machines",
    "No graph generated",
    "Trend-based visualization"
]

def main():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    base_time = datetime.now() - timedelta(days=15)

    records = []

    for i in range(200):
        timestamp = base_time + timedelta(minutes=45 * i)

        fake_table_data = {
            "MachineName": f"Machine_{random.randint(1, 10)}",
            "Value": round(random.uniform(100, 1000), 2)
        }

        record = (
            random.choice(users),             # Username
            random.choice(questions),         # User_Question
            random.choice(explanations),      # Explanation
            random.choice(graphs),            # Graph
            json.dumps(fake_table_data),      # TableData
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            None                               # GraphImage
        )

        records.append(record)

    cur.executemany("""
    INSERT INTO Chat_History
    ("Username","User_Question","Explanation","Graph","TableData","Timestamp","GraphImage")
    VALUES (?,?,?,?,?,?,?)
    """, records)

    conn.commit()
    conn.close()

    print("✅ Successfully inserted", len(records), "chat history rows.")


if __name__ == "__main__":
    main()

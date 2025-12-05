import sqlite3

DB_PATH = "steel.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. Check Min and Max timestamp
cursor.execute("""
SELECT MIN("time"), MAX("time")
FROM "EMS_Daily";
""")
min_time, max_time = cursor.fetchone()
print("Oldest time:", min_time)
print("Newest time:", max_time)

print("-" * 50)

# 2. Show latest 10 timestamps
cursor.execute("""
SELECT "time"
FROM "EMS_Daily"
ORDER BY "time" DESC
LIMIT 10;
""")

print("Last 10 records:")
for row in cursor.fetchall():
    print(row[0])

print("-" * 50)

# 3. Try LIKE test
cursor.execute("""
SELECT AVG("NCH20")
FROM "EMS_Daily"
WHERE "time" LIKE '%2025%';
""")

avg_speed = cursor.fetchone()[0]
print("Average spindle speed for 2025 pattern:", avg_speed)

conn.close()

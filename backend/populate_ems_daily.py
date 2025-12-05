import sqlite3
import random
from datetime import datetime, timedelta

DB_NAME = "steel.db"

plants = ["Jamshed Plant", "Kalinga Plant", "Durgapur Works", "Rourkela Unit"]
subplants = ["Hot Strip Mill", "Cold Rolling Mill", "Blast Furnace Zone", "Rolling Section"]
divisions = ["Hot Rolling", "Cold Rolling", "Iron Making", "Steel Melting"]
processes = ["Rolling", "Annealing", "Casting", "Finishing"]
machines = [
    "CR_Mill_01", "CR_Mill_02", "HR_Mill_01", "BF_Unit_1",
    "BF_Unit_2", "SM_Conv_01", "SM_Conv_02", "RM_Stand_01",
    "RM_Stand_02", "Shear_Line_01"
]

alarms = [
    "Normal", "Overheating Warning", "Spindle Overload", 
    "Hydraulic Pressure Low", "Cooling Failure", "Motor Current Surge"
]

def random_float(a, b):
    return round(random.uniform(a, b), 2)

# === Machine Profiles ===
def generate_machine_profile():
    return {
        "speed_base": random.randint(900, 1400),
        "load_base": random.uniform(20, 70),
        "temp_base": random.uniform(45, 70),
        "current_base": random.uniform(8, 20)
    }

machine_profiles = {m: generate_machine_profile() for m in machines}

# === Row Generator ===
def generate_row(time):
    machine = random.choice(machines)
    p = machine_profiles[machine]
    hour = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").hour

    def time_factor():
        if 6 <= hour <= 10: return 1.1
        if 11 <= hour <= 18: return 1.3
        if 19 <= hour <= 22: return 1.15
        return 0.8
    
    tf = time_factor()

    fault = random.random() < 0.04  # 4% chance instead of 5%
    alarm = random.choice(alarms) if fault else "Normal"

    def jitter(v, scale=0.07):
        return round(v * (tf + random.uniform(-scale, scale)), 2)

    speed = jitter(p["speed_base"], 0.1)
    load = jitter(p["load_base"], 0.2)
    temp = jitter(p["temp_base"] + load * 0.12, 0.05)
    current = jitter(p["current_base"] + load * 0.04, 0.1)

    if fault:
        temp *= random.uniform(1.2, 1.6)
        load *= random.uniform(1.4, 1.9)
        current *= random.uniform(1.3, 1.7)

    return (
        time,
        f"IOT-{random.randint(100, 999)}",
        machine,
        f"MC-{random.randint(1000, 9999)}",
        random.choice(divisions),
        random.choice(plants),
        random.choice(subplants),
        random.choice(processes),

        random_float(12, 24),
        random_float(10, 22),
        random_float(14, 26),
        random_float(15, 28),
        load * 0.8, load * 1.1, load * 1.3, load * 1.2,

        speed, speed * 0.98, speed * 1.02, speed * 1.01,

        temp * 0.95, temp, temp * 1.1, temp * 0.9,

        jitter(150, 0.2),
        current,
        load,
        speed,
        temp,
        random_float(350, 460),
        round(load * 40 * random.uniform(0.85, 1.15), 2),
        1 if fault else 0,
        alarm,
        current * 1.1,
        current * 1.05,
        load * 0.9,
        load * 1.1,
        speed * 1.05
    )


def main():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # ====== Generate data from Sept 1 to Dec 31 ======
    start_date = datetime(2025, 9, 1, 0, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59, 59)

    rows = []
    current = start_date
    while current <= end_date:
        rows.append(generate_row(current.strftime("%Y-%m-%d %H:%M:%S")))
        current += timedelta(minutes=15)

    # === Insert Data ===
    cur.executemany("""
    INSERT INTO EMS_Daily VALUES (
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,?,?,?,?,
        ?,?,?,?,?,?
    )
    """, rows)

    conn.commit()
    conn.close()

    print("✅ Inserted", len(rows), "high-quality rows into EMS_Daily")

if __name__ == "__main__":
    main()

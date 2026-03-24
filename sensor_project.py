import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ══════════════════════════════════════
# STEP 1: Generate Sensor Data
# ══════════════════════════════════════
print("=== STEP 1: Generate Data ===")

np.random.seed(42)
n = 200

# Normal operation
temp_normal      = np.random.normal(70, 3, n)
vibration_normal = np.random.normal(0.5, 0.1, n)
pressure_normal  = np.random.normal(100, 5, n)

# Fault readings (20 samples)
temp_fault      = np.random.normal(95, 5, 20)
vibration_fault = np.random.normal(2.5, 0.5, 20)
pressure_fault  = np.random.normal(140, 10, 20)

# Combine
temperature = np.concatenate([temp_normal, temp_fault])
vibration   = np.concatenate([vibration_normal, vibration_fault])
pressure    = np.concatenate([pressure_normal, pressure_fault])

df = pd.DataFrame({
    "temperature": temperature,
    "vibration":   vibration,
    "pressure":    pressure
})

print(f"Total samples: {len(df)}")
print(df.describe().round(2))

# ══════════════════════════════════════
# STEP 2: Train Model
# ══════════════════════════════════════
print("\n=== STEP 2: Model Training ===")

model = IsolationForest(contamination=0.1, random_state=42)
df["status"] = model.fit_predict(df[["temperature", "vibration", "pressure"]])
df["status"]  = df["status"].map({1: "Normal", -1: "FAULT"})

print(df["status"].value_counts())

# ══════════════════════════════════════
# STEP 3: Visualize
# ══════════════════════════════════════
print("\n=== STEP 3: Visualization ===")

colors = df["status"].map({"Normal": "blue", "FAULT": "red"})

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df["temperature"], df["vibration"], c=colors, alpha=0.6)
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration")
plt.title("Temperature vs Vibration")

plt.subplot(1, 2, 2)
plt.scatter(df["temperature"], df["pressure"], c=colors, alpha=0.6)
plt.xlabel("Temperature (°C)")
plt.ylabel("Pressure")
plt.title("Temperature vs Pressure")

plt.tight_layout()
plt.savefig("sensor_chart.png")
print("Chart saved: sensor_chart.png ✅")

# ══════════════════════════════════════
# STEP 4: Results
# ══════════════════════════════════════
print("\n=== STEP 4: Results ===")
faults = df[df["status"] == "FAULT"]
normal = df[df["status"] == "Normal"]

print(f"Total faults detected:      {len(faults)}")
print(f"Average fault temperature:  {faults['temperature'].mean():.1f}°C")
print(f"Average normal temperature: {normal['temperature'].mean():.1f}°C")
print(f"Detection rate:             {len(faults)/20*100:.0f}%")
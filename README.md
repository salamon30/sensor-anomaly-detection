# Sensor Anomaly Detection

A machine learning project that detects equipment failures using sensor data — before they happen.

## What It Does
- Monitors 3 sensor readings: **temperature**, **vibration**, and **pressure**
- Detects anomalies using **Isolation Forest** algorithm
- Visualizes normal vs faulty readings on a scatter plot

## Results
- Analyzed **220 sensor readings**
- Detected **22 fault conditions**
- Average fault temperature: **93.6°C** vs normal **69.9°C**

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
```bash
pip install numpy pandas scikit-learn matplotlib
python3 sensor_project.py
```

## Use Cases
This type of anomaly detection is used in:
- Industrial automation (Siemens, ABB)
- Medical devices (Roche, Novartis)
- Automotive systems (BMW, Bosch)

## Author
Recep Uzun — AI Master's Student @ Deggendorf Institute of Technology

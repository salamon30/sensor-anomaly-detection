"""
Sensor Anomaly Detection using Isolation Forest.

Detects equipment faults from multivariate sensor readings
(temperature, vibration, pressure) without labelled training data.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "n_normal": 200,
    "n_faults": 20,
    "contamination": 0.09,
    "random_seed": 42,
    "output_path": "sensor_chart.png",
}

COLORS = {"Normal": "#2196F3", "FAULT": "#F44336"}

# ─── Data Generation ──────────────────────────────────────────────────────────

def generate_sensor_data(
    n_normal: int, n_faults: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Simulate sensor readings for normal operation and fault conditions."""
    rng = np.random.default_rng(seed)

    normal = {
        "temperature": rng.normal(70, 3, n_normal),
        "vibration":   rng.normal(0.5, 0.1, n_normal),
        "pressure":    rng.normal(100, 5, n_normal),
    }
    fault = {
        "temperature": rng.normal(95, 5, n_faults),
        "vibration":   rng.normal(2.5, 0.5, n_faults),
        "pressure":    rng.normal(140, 10, n_faults),
    }

    df = pd.DataFrame({k: np.concatenate([normal[k], fault[k]]) for k in normal})
    df.index.name = "reading_id"

    ground_truth = np.array(["Normal"] * n_normal + ["FAULT"] * n_faults)
    return df, ground_truth


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame, contamination: float, seed: int
) -> Tuple[pd.DataFrame, IsolationForest]:
    """Scale features and run Isolation Forest to flag anomalous readings."""
    features = ["temperature", "vibration", "pressure"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=seed,
    )
    predictions = model.fit_predict(X)

    result = df.copy()
    result["anomaly_score"] = model.score_samples(X)
    result["status"] = pd.Series(predictions).map({1: "Normal", -1: "FAULT"}).values
    return result, model


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualize(df: pd.DataFrame, output_path: str) -> None:
    """Render a 4-panel diagnostic dashboard and save to disk."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "gray",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    c = df["status"].map(COLORS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Sensor Anomaly Detection  —  Isolation Forest",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # Time-series: Temperature
    ax = axes[0, 0]
    ax.scatter(df.index, df["temperature"], c=c, s=16, alpha=0.75)
    ax.set_xlabel("Reading ID"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Over Time")

    # Time-series: Vibration
    ax = axes[0, 1]
    ax.scatter(df.index, df["vibration"], c=c, s=16, alpha=0.75)
    ax.set_xlabel("Reading ID"); ax.set_ylabel("Vibration")
    ax.set_title("Vibration Over Time")

    # Scatter: Temperature vs Pressure
    ax = axes[1, 0]
    ax.scatter(df["temperature"], df["pressure"], c=c, s=18, alpha=0.75)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Pressure")
    ax.set_title("Temperature vs Pressure")

    # Anomaly Score Distribution
    ax = axes[1, 1]
    for label, grp in df.groupby("status"):
        ax.hist(grp["anomaly_score"], bins=22, alpha=0.65,
                label=label, color=COLORS[label], edgecolor="white")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend(framealpha=0.8)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["Normal"], markersize=9, label="Normal"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["FAULT"], markersize=9, label="FAULT"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=11, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Dashboard saved → {output_path}")


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, n_faults: int) -> None:
    """Print a precision/recall summary against known ground truth."""
    faults  = df[df["status"] == "FAULT"]
    normals = df[df["status"] == "Normal"]

    detected   = set(df.index[df["status"] == "FAULT"])
    true_faults = set(range(len(df) - n_faults, len(df)))

    tp = len(detected & true_faults)
    fp = len(detected - true_faults)
    fn = len(true_faults - detected)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "═" * 52)
    print("   ANOMALY DETECTION REPORT")
    print("═" * 52)
    print(f"   Total readings      : {len(df)}")
    print(f"   Faults detected     : {len(faults)}  (ground truth: {n_faults})")
    print(f"   Normal readings     : {len(normals)}")
    print("─" * 52)
    print(f"   Avg fault temp      : {faults['temperature'].mean():.1f} °C")
    print(f"   Avg normal temp     : {normals['temperature'].mean():.1f} °C")
    print(f"   Avg fault pressure  : {faults['pressure'].mean():.1f}")
    print("─" * 52)
    print(f"   Precision           : {precision:.1%}")
    print(f"   Recall              : {recall:.1%}")
    print(f"   F1 Score            : {f1:.1%}")
    print("═" * 52)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect sensor anomalies using Isolation Forest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-normal",      type=int,   default=CONFIG["n_normal"],      help="Normal reading count")
    parser.add_argument("--n-faults",      type=int,   default=CONFIG["n_faults"],      help="Fault reading count")
    parser.add_argument("--contamination", type=float, default=CONFIG["contamination"], help="Expected fault ratio (0–0.5)")
    parser.add_argument("--seed",          type=int,   default=CONFIG["random_seed"],   help="Random seed")
    parser.add_argument("--output",        type=str,   default=CONFIG["output_path"],   help="Output chart path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[ 1/3 ]  Generating sensor data ...")
    df, ground_truth = generate_sensor_data(args.n_normal, args.n_faults, args.seed)
    print(f"         {len(df)} readings  ({args.n_normal} normal + {args.n_faults} fault)\n")

    print("[ 2/3 ]  Running Isolation Forest ...")
    df, _ = detect_anomalies(df, args.contamination, args.seed)
    print(f"         {df['status'].value_counts().to_dict()}\n")

    print("[ 3/3 ]  Rendering diagnostic dashboard ...")
    visualize(df, args.output)

    print_report(df, args.n_faults)


if __name__ == "__main__":
    main()

"""
occupancy_model.py
------------------
Neural-network add-on (spec requirement).

Generates synthetic training data then trains an MLPRegressor to predict
parking occupancy given 5 features:
    hour          : 0-23
    weekday       : 0 (Mon) – 6 (Sun)
    lot_id        : integer-encoded lot (0-4)
    class_density : estimated concurrent classes nearby (0-10)
    event         : campus event flag (0 or 1)

Output: occupancy fraction 0.0 – 1.0
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from campus_graph import PARKING_LOTS, LOT_ID_MAP

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

# Hourly busyness weights (0=midnight … 23=11 pm)
HOUR_WEIGHT = np.array([
    0.05, 0.03, 0.02, 0.02, 0.03, 0.05,   # 00-05
    0.10, 0.30, 0.75, 0.90, 0.95, 0.92,   # 06-11
    0.85, 0.88, 0.87, 0.82, 0.70, 0.50,   # 12-17
    0.35, 0.25, 0.18, 0.12, 0.08, 0.06,   # 18-23
])

# Mon-Thu busiest, Fri lighter, weekend very quiet
WEEKDAY_WEIGHT = np.array([1.0, 1.0, 1.0, 0.95, 0.80, 0.30, 0.15])

# Base popularity per lot
LOT_BASE = {
    "P_NORTH":   0.82,
    "P_EAST":    0.70,
    "P_CENTRAL": 0.95,
    "P_WEST":    0.60,
    "P_SOUTH":   0.55,
}

FEATURE_COLS = ["hour", "weekday", "lot_id", "class_density", "event"]
TARGET_COL   = "occupancy_pct"


def _class_density(hour, weekday, rng):
    """Estimate concurrent classes at a given hour."""
    if weekday >= 5:
        return 0
    if 8 <= hour <= 17:
        return int(rng.integers(4, 11))
    if 18 <= hour <= 20:
        return int(rng.integers(0, 4))
    return 0


def generate_data(n_days=180, seed=42):
    """
    Simulate n_days of hourly occupancy across all parking lots.

    Uses pandas DataFrame as the data structure — each row is one
    observation: (hour, weekday, lot_id, class_density, event, occupancy_pct).

    Returns X (numpy features array) and y (numpy target array)
    extracted from the DataFrame, ready for scikit-learn.
    """
    rng     = np.random.default_rng(seed)
    records = []   # list of dicts — pandas builds a DataFrame from this

    for day in range(n_days):
        weekday = day % 7
        event   = int(weekday < 5 and rng.random() < 0.15)

        for hour in range(24):
            cd = _class_density(hour, weekday, rng)
            hw = HOUR_WEIGHT[hour]
            ww = WEEKDAY_WEIGHT[weekday]
            event_boost = 0.15 if (event and 10 <= hour <= 20) else 0.0

            for lot_name, lot_id in LOT_ID_MAP.items():
                base = LOT_BASE[lot_name]
                occ  = base * hw * ww + event_boost + rng.normal(0, 0.06)
                occ  = float(np.clip(occ, 0.0, 1.0))

                records.append({
                    "hour":          hour,
                    "weekday":       weekday,
                    "lot_id":        lot_id,
                    "class_density": cd,
                    "event":         event,
                    "occupancy_pct": round(occ, 4),
                })

    # Build the pandas DataFrame — this is the "simulated occupancy data" input
    df = pd.DataFrame(records)

    # Extract features and target as numpy arrays for scikit-learn
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y, df   # also return df so callers can inspect / display it


# ---------------------------------------------------------------------------
# OccupancyPredictor
# ---------------------------------------------------------------------------

class OccupancyPredictor:
    """
    Wraps a trained MLPRegressor + StandardScaler.

    Usage:
        predictor = OccupancyPredictor()
        predictor.train()
        occ = predictor.predict("P_CENTRAL", hour=9, weekday=1, event=0)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self._trained = False

    def train(self, n_days=180, verbose=True):
        """Train on synthetic data. Returns evaluation metrics dict."""
        X, y, df = generate_data(n_days=n_days)

        if verbose:
            # Use pandas to summarise the generated dataset before training
            print(f"  [Data] Simulated {len(df):,} occupancy observations (pandas DataFrame)")
            print(f"         Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"         Columns: {list(df.columns)}")
            lot_id_to_name = {v: k for k, v in LOT_ID_MAP.items()}
            summary = df.groupby("lot_id")["occupancy_pct"].mean()
            print("         Average occupancy per lot:")
            for lot_id, mean_occ in summary.items():
                name = lot_id_to_name.get(lot_id, str(lot_id))
                print(f"           {name:<12}: {mean_occ:.1%}")
            print()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model.fit(X_train_s, y_train)
        self._trained = True

        y_pred = np.clip(self.model.predict(X_test_s), 0, 1)
        metrics = {
            "MAE":            round(float(mean_absolute_error(y_test, y_pred)), 4),
            "R2":             round(float(r2_score(y_test, y_pred)), 4),
            "samples_train":  len(X_train),
            "samples_test":   len(X_test),
        }

        if verbose:
            print("  [ML] Neural network training complete.")
            print(f"       MAE : {metrics['MAE']}  (avg error in occupancy fraction, lower = better)")
            print(f"       R²  : {metrics['R2']}  (1.0 = perfect, measures how well model fits data)")
            print(f"       Data: {metrics['samples_train']} train / {metrics['samples_test']} test samples")

        return metrics

    def _auto_class_density(self, hour, weekday):
        if weekday >= 5:
            return 0
        return 7 if 8 <= hour <= 17 else (2 if 18 <= hour <= 20 else 0)

    def predict(self, lot_name, hour, weekday, event=0, class_density=None):
        """
        Predict occupancy fraction (0.0–1.0) for one lot.

        Parameters
        ----------
        lot_name      : e.g. 'P_CENTRAL'
        hour          : 0-23
        weekday       : 0 (Mon) – 6 (Sun)
        event         : 1 if campus event today, else 0
        class_density : auto-estimated if None
        """
        if not self._trained:
            raise RuntimeError("Call .train() first.")
        if class_density is None:
            class_density = self._auto_class_density(hour, weekday)

        feat = np.array([[hour, weekday, LOT_ID_MAP[lot_name], class_density, event]])
        occ  = float(np.clip(self.model.predict(self.scaler.transform(feat))[0], 0.0, 1.0))
        return round(occ, 4)

    def predict_all(self, hour, weekday, event=0):
        """Return {lot_id: occupancy_fraction} for every parking lot."""
        return {lot: self.predict(lot, hour, weekday, event) for lot in LOT_ID_MAP}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = OccupancyPredictor()
    p.train()
    print("\nPredictions — Monday 9 AM, no event:")
    for lot, occ in sorted(p.predict_all(9, 0).items()):
        free = round(PARKING_LOTS[lot]["capacity"] * (1 - occ))
        print(f"  {lot:12s}: {occ:.1%} occupied  ({free} free)")

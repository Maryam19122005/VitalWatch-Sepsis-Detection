import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import json

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')
os.makedirs('models', exist_ok=True)

print("Loading data for Time Series analysis...")
df = pd.read_sql("""
    SELECT patient_id, iculos, hr, sbp, temp, resp, o2sat 
    FROM patient_features 
    ORDER BY patient_id, iculos
""", engine)
print(f"Loaded {len(df):,} rows")

# ── APPROACH: Use last N hours to predict next hour ────────────────────────
# We use a sliding window approach - simple and course appropriate
WINDOW_SIZE = 6   # use last 6 hours
TARGET_COL  = 'hr'

print(f"\nBuilding sliding window sequences (window={WINDOW_SIZE} hours)...")

# Get patients with enough hours for meaningful time series
patient_counts = df.groupby('patient_id').size()
valid_patients  = patient_counts[patient_counts >= WINDOW_SIZE + 1].index
df_valid = df[df['patient_id'].isin(valid_patients)]
print(f"Patients with >={WINDOW_SIZE+1} hours: {len(valid_patients):,}")

# Build X (last 6 HR values) → y (next HR value)
X_sequences = []
y_targets   = []

for patient_id, group in df_valid.groupby('patient_id'):
    hr_values = group[TARGET_COL].values
    
    for i in range(len(hr_values) - WINDOW_SIZE):
        window = hr_values[i:i + WINDOW_SIZE]
        target = hr_values[i + WINDOW_SIZE]
        X_sequences.append(window)
        y_targets.append(target)

X_sequences = np.array(X_sequences)
y_targets   = np.array(y_targets)
print(f"Total sequences created: {len(X_sequences):,}")

# ── Train/Test Split ───────────────────────────────────────────────────────
split = int(len(X_sequences) * 0.8)
X_train = X_sequences[:split]
X_test  = X_sequences[split:]
y_train = y_targets[:split]
y_test  = y_targets[split:]

# ── MODEL 1: ARIMA-style (using linear regression on window) ──────────────
print("\n" + "="*55)
print("TIME SERIES MODEL 1: Linear Trend (ARIMA baseline)")
print("="*55)

from sklearn.linear_model import LinearRegression

# Create time indices as features (simulating ARIMA trend component)
time_indices = np.arange(WINDOW_SIZE).reshape(1, -1)
X_train_lr   = X_train  # each window is the feature
X_test_lr    = X_test

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train)
lr_preds = lr_model.predict(X_test_lr)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae  = mean_absolute_error(y_test, lr_preds)
print(f"  RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f}")

# ── MODEL 2: Exponential Smoothing (weighted recent values more) ───────────
print("\n" + "="*55)
print("TIME SERIES MODEL 2: Exponential Smoothing")
print("="*55)

# Simple exponential smoothing - weight recent hours more
alpha = 0.3  # smoothing factor
weights = np.array([(1-alpha)**i for i in range(WINDOW_SIZE-1, -1, -1)])
weights = weights / weights.sum()  # normalize

es_preds = np.dot(X_test, weights)
es_rmse  = np.sqrt(mean_squared_error(y_test, es_preds))
es_mae   = mean_absolute_error(y_test, es_preds)
print(f"  Alpha (smoothing): {alpha}")
print(f"  RMSE: {es_rmse:.4f} | MAE: {es_mae:.4f}")

# ── MODEL 3: XGBoost for Time Series ──────────────────────────────────────
print("\n" + "="*55)
print("TIME SERIES MODEL 3: XGBoost Forecaster")
print("="*55)

from xgboost import XGBRegressor

# Add extra features: mean, std, trend of window
def add_window_features(X):
    means  = X.mean(axis=1, keepdims=True)
    stds   = X.std(axis=1,  keepdims=True)
    trends = (X[:, -1] - X[:, 0]).reshape(-1, 1)  # last - first = trend
    return np.hstack([X, means, stds, trends])

X_train_xgb = add_window_features(X_train)
X_test_xgb  = add_window_features(X_test)

xgb_ts = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_ts.fit(X_train_xgb, y_train)
xgb_preds = xgb_ts.predict(X_test_xgb)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_mae  = mean_absolute_error(y_test, xgb_preds)
print(f"  RMSE: {xgb_rmse:.4f} | MAE: {xgb_mae:.4f}")

# ── Pick best time series model ────────────────────────────────────────────
results = {
    'Linear Trend': {'RMSE': lr_rmse,  'MAE': lr_mae,  'model': lr_model},
    'Exp Smoothing': {'RMSE': es_rmse, 'MAE': es_mae,  'model': None},
    'XGBoost TS':   {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'model': xgb_ts},
}

best_ts_name  = min(results, key=lambda x: results[x]['RMSE'])
best_ts_rmse  = results[best_ts_name]['RMSE']
best_ts_model = results[best_ts_name]['model']

print(f"\n  Best Time Series Model: {best_ts_name} (RMSE={best_ts_rmse:.4f})")

if best_ts_model is not None:
    joblib.dump(best_ts_model, 'models/best_timeseries.pkl')
    print("  Saved -> models/best_timeseries.pkl")

# Save metadata
ts_results = {
    'window_size': WINDOW_SIZE,
    'target_column': TARGET_COL,
    'total_sequences': len(X_sequences),
    'best_model': best_ts_name,
    'best_rmse': round(float(best_ts_rmse), 4),
    'all_models': {
        name: {'RMSE': round(float(v['RMSE']), 4), 
               'MAE':  round(float(v['MAE']),  4)}
        for name, v in results.items()
    }
}

with open('models/timeseries_results.json', 'w') as f:
    json.dump(ts_results, f, indent=2)

print("\n" + "="*55)
print("TIME SERIES COMPLETE!")
print("="*55)
for name, v in results.items():
    print(f"  {name:20s} -> RMSE={v['RMSE']:.4f} | MAE={v['MAE']:.4f}")
print(f"\n  Best: {best_ts_name} | RMSE={best_ts_rmse:.4f}")
print("="*55)
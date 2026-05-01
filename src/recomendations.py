import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')
os.makedirs('models', exist_ok=True)

print("Loading data for Recommendation System...")
df = pd.read_sql("""
    SELECT patient_id,
           AVG(hr) as avg_hr, AVG(temp) as avg_temp,
           AVG(o2sat) as avg_o2sat, AVG(sbp) as avg_sbp,
           AVG(map) as avg_map, AVG(resp) as avg_resp,
           MAX(qsofa) as max_qsofa,
           MAX(flag_tachy) as had_tachy,
           MAX(flag_fever) as had_fever,
           MAX(flag_low_bp) as had_low_bp,
           MAX(flag_hypoxia) as had_hypoxia,
           MAX(sepsislabel) as had_sepsis,
           MAX(iculos) as total_icu_hours
    FROM patient_features
    GROUP BY patient_id
""", engine)

print(f"Patient profiles loaded: {len(df):,}")

SIMILARITY_FEATURES = [
    'avg_hr', 'avg_temp', 'avg_o2sat', 'avg_sbp', 'avg_map',
    'avg_resp', 'max_qsofa', 'had_tachy', 'had_fever',
    'had_low_bp', 'had_hypoxia', 'total_icu_hours'
]

X = df[SIMILARITY_FEATURES].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── KNN with K=11 ──────────────────────────────────────────────────────────
# K=11 because we skip index 0 (patient itself)
# so we always get exactly 10 real matches
print("\nTraining KNN Recommendation Model (K=11)...")
knn = NearestNeighbors(n_neighbors=11, metric='euclidean', n_jobs=-1)
knn.fit(X_scaled)
print(f"KNN indexed {len(df):,} patient profiles")
print(f"Each query will return top 10 most similar patients")

# ── Test with a high risk patient ──────────────────────────────────────────
print("\n" + "="*55)
print("TESTING RECOMMENDATION SYSTEM")
print("="*55)

new_patient = pd.DataFrame([{
    'avg_hr':         105,
    'avg_temp':       38.5,
    'avg_o2sat':      91,
    'avg_sbp':        88,
    'avg_map':        62,
    'avg_resp':       24,
    'max_qsofa':      2,
    'had_tachy':      1,
    'had_fever':      1,
    'had_low_bp':     1,
    'had_hypoxia':    1,
    'total_icu_hours': 8
}])

new_patient_scaled = scaler.transform(new_patient)
distances, indices = knn.kneighbors(new_patient_scaled)

# Skip index 0 → gives us 10 real matches
similar_patients = df.iloc[indices[0][1:]]
similar_distances = distances[0][1:]

print(f"\nNew Patient Profile:")
print(f"  HR={105} bpm      | Temp={38.5}C")
print(f"  O2Sat={91}%       | MAP={62} mmHg")
print(f"  Resp Rate={24}    | qSOFA={2}")
print(f"  Flags: Tachycardia, Fever, LowBP, Hypoxia")

print(f"\nTop 10 Most Similar Historical Patients:")
print(f"{'#':<4} {'Patient':<12} {'HR':>6} {'Temp':>6} "
      f"{'O2Sat':>6} {'ICU Hrs':>8} {'Outcome':<15} {'Similarity':>10}")
print("-" * 75)

for i, (idx, row) in enumerate(similar_patients.iterrows()):
    outcome    = "HAD SEPSIS" if row['had_sepsis'] == 1 else "No Sepsis"
    distance   = similar_distances[i]
    similarity = round(1 / (1 + distance) * 100, 1)  # convert to % score
    print(f"{i+1:<4} {row['patient_id']:<12} "
          f"{row['avg_hr']:>6.0f} "
          f"{row['avg_temp']:>6.1f} "
          f"{row['avg_o2sat']:>6.1f} "
          f"{row['total_icu_hours']:>8.0f} "
          f"{outcome:<15} "
          f"{similarity:>9.1f}%")

# ── Risk Assessment ────────────────────────────────────────────────────────
sepsis_count    = int(similar_patients['had_sepsis'].sum())
sepsis_risk_pct = round(sepsis_count / 10 * 100)
avg_icu_hours   = similar_patients['total_icu_hours'].mean()

print(f"\n{'='*55}")
print(f"RISK ASSESSMENT SUMMARY")
print(f"{'='*55}")
print(f"  Similar patients analyzed : 10")
print(f"  Developed sepsis          : {sepsis_count} / 10")
print(f"  Estimated sepsis risk     : {sepsis_risk_pct}%")
print(f"  Avg ICU stay (similar)    : {avg_icu_hours:.0f} hours")

# Updated thresholds - more clinically sensitive
if sepsis_count >= 5:
    risk_level     = "HIGH RISK"
    recommendation = "Initiate sepsis protocol immediately"
    action         = "Order blood cultures, lactate, start IV antibiotics"
elif sepsis_count >= 2:
    risk_level     = "MODERATE RISK"
    recommendation = "Increase monitoring frequency to every 30 minutes"
    action         = "Repeat vitals hourly, order CBC and metabolic panel"
else:
    risk_level     = "LOW RISK"
    recommendation = "Continue standard monitoring"
    action         = "Routine vitals every 4 hours"

print(f"\n  Risk Level      : {risk_level}")
print(f"  Recommendation  : {recommendation}")
print(f"  Suggested Action: {action}")

# ── Save everything ────────────────────────────────────────────────────────
joblib.dump(knn,    'models/knn_recommendation.pkl')
joblib.dump(scaler, 'models/recommendation_scaler.pkl')
joblib.dump(df,     'models/patient_profiles.pkl')

results = {
    'total_patients_indexed': len(df),
    'k_neighbors': 10,
    'similarity_features': SIMILARITY_FEATURES,
    'metric': 'euclidean',
    'risk_thresholds': {
        'HIGH':     '5+ out of 10 similar patients had sepsis',
        'MODERATE': '2-4 out of 10 similar patients had sepsis',
        'LOW':      '0-1 out of 10 similar patients had sepsis'
    },
    'test_result': {
        'sepsis_count': sepsis_count,
        'risk_pct': sepsis_risk_pct,
        'risk_level': risk_level,
        'recommendation': recommendation
    }
}

with open('models/recommendation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*55}")
print(f"RECOMMENDATION SYSTEM COMPLETE!")
print(f"  {len(df):,} patients indexed")
print(f"  K=10 nearest neighbors")
print(f"  Saved -> models/knn_recommendation.pkl")
print(f"{'='*55}")

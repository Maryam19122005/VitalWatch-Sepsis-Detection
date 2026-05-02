import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("="*55)
print("VITALWATCH - ML PIPELINE TESTS (DeepChecks)")
print("="*55)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PASSED = []
FAILED = []

def test(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name}")
        PASSED.append(name)
    else:
        print(f"  FAIL: {name} {detail}")
        FAILED.append(name)

# ── 1. Model Files Exist ───────────────────────────────
print("\n[1] Model File Checks")
required_files = [
    'best_classifier.pkl',
    'best_regressor.pkl',
    'best_timeseries.pkl',
    'kmeans_clustering.pkl',
    'knn_recommendation.pkl',
    'pca_reducer.pkl',
    'scaler.pkl',
    'recommendation_scaler.pkl',
    'cluster_risk_labels.pkl',
    'patient_profiles.pkl',
    'manifest.json',
    'association_rules.json'
]
for f in required_files:
    path = os.path.join(MODELS_DIR, f)
    test(f"File exists: {f}", os.path.exists(path))

# ── 2. Models Load Without Error ──────────────────────
print("\n[2] Model Loading Checks")
try:
    clf     = joblib.load(os.path.join(MODELS_DIR, 'best_classifier.pkl'))
    reg     = joblib.load(os.path.join(MODELS_DIR, 'best_regressor.pkl'))
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    kmeans  = joblib.load(os.path.join(MODELS_DIR, 'kmeans_clustering.pkl'))
    pca     = joblib.load(os.path.join(MODELS_DIR, 'pca_reducer.pkl'))
    test("Classifier loads", clf is not None)
    test("Regressor loads",  reg is not None)
    test("Scaler loads",     scaler is not None)
    test("KMeans loads",     kmeans is not None)
    test("PCA loads",        pca is not None)
except Exception as e:
    test("All models load", False, str(e))

# ── 3. Manifest Checks ─────────────────────────────────
print("\n[3] Manifest Validation")
try:
    with open(os.path.join(MODELS_DIR, 'manifest.json')) as f:
        manifest = json.load(f)
    test("Manifest has classification",
         'classification' in manifest)
    test("Manifest has regression",
         'regression' in manifest)
    test("Best model recorded",
         bool(manifest.get('classification', {}).get('best_model')))
    test("Medical score > 0.5",
         manifest.get('classification', {})
                 .get('best_medical_score', 0) > 0.5)
    test("Regression RMSE < 15",
         manifest.get('regression', {})
                 .get('best_rmse', 999) < 15)
    test("Train rows > 500000",
         manifest.get('train_rows', 0) > 500000)
except Exception as e:
    test("Manifest readable", False, str(e))

# ── 4. Model Output Range Checks ──────────────────────
print("\n[4] Model Output Range Checks")
try:
    FEATURE_COLS = [
        'hr','o2sat','temp','sbp','map','dbp','resp',
        'age','gender',
        'hr_mean_3h','hr_mean_6h','sbp_mean_3h',
        'map_mean_3h','resp_mean_3h',
        'hr_trend','sbp_trend','temp_trend','resp_trend',
        'hr_std_3h','sbp_std_3h',
        'flag_tachy','flag_hypoxia','flag_fever',
        'flag_hypotemp','flag_low_bp','flag_tachypnea',
        'qsofa','hours_in_icu'
    ]

    # Normal patient
    normal = dict(zip(FEATURE_COLS,
        [75, 97, 37.0, 120, 80, 70, 16, 50, 1,
         75, 74, 120, 80, 16,
         0, 0, 0, 0, 2, 2,
         0, 0, 0, 0, 0, 0, 0, 4]))

    # High risk patient
    high_risk = dict(zip(FEATURE_COLS,
        [115, 88, 39.2, 85, 58, 45, 28, 72, 1,
         110, 105, 90, 62, 26,
         5, -8, 0.5, 3, 8, 10,
         1, 1, 1, 0, 1, 1, 3, 12]))

    X_normal    = scaler.transform([list(normal.values())])
    X_high_risk = scaler.transform([list(high_risk.values())])

    # Classification output range
    prob_normal    = float(clf.predict_proba(X_normal)[0][1])
    prob_high_risk = float(clf.predict_proba(X_high_risk)[0][1])
    test("Normal patient prob 0-1",
         0.0 <= prob_normal <= 1.0)
    test("High risk patient prob 0-1",
         0.0 <= prob_high_risk <= 1.0)
    test("High risk prob > normal prob",
         prob_high_risk > prob_normal,
         f"high={prob_high_risk:.3f} normal={prob_normal:.3f}")

    # Regression output range
    hr_pred = float(reg.predict(X_normal)[0])
    test("Regression output is number",
         isinstance(hr_pred, float))
    test("Regression HR in valid range (30-250)",
         30 <= hr_pred <= 250,
         f"Got {hr_pred:.1f}")

    # Clustering output range
    cluster_id = int(kmeans.predict(X_normal)[0])
    test("Cluster ID between 0-3",
         0 <= cluster_id <= 3,
         f"Got {cluster_id}")

    # PCA output shape
    coords = pca.transform(X_normal)[0]
    test("PCA returns 2 components",
         len(coords) == 2)

    print(f"\n    Normal patient:    prob={prob_normal:.3f}, "
          f"cluster={cluster_id}, hr_pred={hr_pred:.1f}")
    print(f"    High risk patient: prob={prob_high_risk:.3f}")

except Exception as e:
    test("Model predictions work", False, str(e))

# ── 5. Data Quality Checks ─────────────────────────────
print("\n[5] Data Quality Checks")
try:
    from sqlalchemy import create_engine
    engine = create_engine(
        'postgresql://postgres:1234@localhost:5432/vitalwatch_db'
    )

    # Check tables exist
    tables = pd.read_sql("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public'
    """, engine)
    table_names = tables['table_name'].tolist()
    test("patient_vitals table exists",
         'patient_vitals' in table_names)
    test("patient_features table exists",
         'patient_features' in table_names)

    # Check row counts
    vitals_count = pd.read_sql(
        "SELECT COUNT(*) as c FROM patient_vitals", engine
    ).iloc[0]['c']
    features_count = pd.read_sql(
        "SELECT COUNT(*) as c FROM patient_features", engine
    ).iloc[0]['c']

    test("patient_vitals has 700K+ rows",
         vitals_count > 700000,
         f"Got {vitals_count:,}")
    test("patient_features has 700K+ rows",
         features_count > 700000,
         f"Got {features_count:,}")

    # Check sepsis label distribution
    dist = pd.read_sql("""
        SELECT sepsislabel, COUNT(*) as cnt
        FROM patient_features
        GROUP BY sepsislabel
    """, engine)
    test("Both classes present (0 and 1)",
         len(dist) == 2)

    # Check no nulls in vital columns
    null_check = pd.read_sql("""
        SELECT COUNT(*) as nulls FROM patient_features
        WHERE hr IS NULL OR temp IS NULL OR o2sat IS NULL
    """, engine).iloc[0]['nulls']
    test("No nulls in vital columns",
         null_check == 0,
         f"Found {null_check} nulls")

    # Check engineered features exist
    cols = pd.read_sql("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='patient_features'
    """, engine)['column_name'].tolist()
    test("hr_mean_3h feature exists",
         'hr_mean_3h' in cols)
    test("qsofa feature exists",
         'qsofa' in cols)
    test("flag_tachy feature exists",
         'flag_tachy' in cols)
    test("Total features >= 60",
         len(cols) >= 60,
         f"Got {len(cols)}")

except Exception as e:
    test("Database checks", False, str(e))

# ── 6. Association Rules Checks ────────────────────────
print("\n[6] Association Rules Checks")
try:
    with open(os.path.join(MODELS_DIR, 'association_rules.json')) as f:
        rules = json.load(f)
    test("Rules file has content",
         len(rules) > 0)
    test("Total rules > 50",
         rules.get('total_rules', 0) > 50,
         f"Got {rules.get('total_rules',0)}")
    test("Has top rules list",
         len(rules.get('top_10_by_lift', [])) > 0)
except Exception as e:
    test("Association rules checks", False, str(e))

# ── 7. API Endpoint Checks ─────────────────────────────
print("\n[7] API Availability Checks")
try:
    import requests
    r = requests.get('http://localhost:8000/health', timeout=5)
    test("API health endpoint responds",
         r.status_code == 200)
    test("API returns operational status",
         r.json().get('status') == 'operational')

    # Test prediction endpoint
    payload = dict(zip(FEATURE_COLS,
        [85, 95, 37.2, 118, 78, 68, 18, 55, 1,
         84, 83, 118, 78, 18,
         0.5, -0.5, 0.1, 0.2, 2.1, 2.8,
         0, 0, 0, 0, 0, 0, 0, 3]))
    r2 = requests.post(
        'http://localhost:8000/api/classify',
        json=payload, timeout=5
    )
    test("Classification endpoint responds",
         r2.status_code == 200)
    test("Response has probability field",
         'probability' in r2.json())

except ImportError:
    test("requests library available", False,
         "pip install requests")
except Exception as e:
    test("API running on port 8000", False,
         "Start API first: uvicorn api.main:app --port 8000")

# ── FINAL SUMMARY ──────────────────────────────────────
print("\n" + "="*55)
print("TEST SUMMARY")
print("="*55)
print(f"  PASSED : {len(PASSED)}")
print(f"  FAILED : {len(FAILED)}")
print(f"  TOTAL  : {len(PASSED) + len(FAILED)}")
print(f"  SCORE  : {len(PASSED)}/{len(PASSED)+len(FAILED)}")

if FAILED:
    print(f"\n  Failed tests:")
    for f in FAILED:
        print(f"    - {f}")
else:
    print("\n  ALL TESTS PASSED!")

print("="*55)

# Exit with error code if tests failed
# This is important for GitHub Actions
if FAILED:
    sys.exit(1)
else:
    sys.exit(0)
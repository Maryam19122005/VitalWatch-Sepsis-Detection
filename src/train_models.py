import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             mean_squared_error, mean_absolute_error, 
                             r2_score, classification_report)
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')
os.makedirs('models', exist_ok=True)

print("Loading features from DB...")
df = pd.read_sql("SELECT * FROM patient_features", engine)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

FEATURE_COLS = [
    'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
    'age', 'gender',
    'hr_mean_3h', 'hr_mean_6h', 'sbp_mean_3h', 'map_mean_3h', 'resp_mean_3h',
    'hr_trend', 'sbp_trend', 'temp_trend', 'resp_trend',
    'hr_std_3h', 'sbp_std_3h',
    'flag_tachy', 'flag_hypoxia', 'flag_fever', 'flag_hypotemp',
    'flag_low_bp', 'flag_tachypnea', 'qsofa', 'hours_in_icu'
]

TARGET_CLASS = 'sepsislabel'
# Fix: predict RAW future HR (iculos+1 shift) instead of rolling average
# We shift hr by -1 to mean "what will HR be next hour"
df['hr_next_hour'] = df.groupby('patient_id')['hr'].shift(-1)
df = df.dropna(subset=['hr_next_hour'])
TARGET_REG = 'hr_next_hour'

X = df[FEATURE_COLS].fillna(0)
y_class = df[TARGET_CLASS]
y_reg   = df[TARGET_REG]

# Calculate imbalance ratio for XGBoost
neg = (y_class == 0).sum()
pos = (y_class == 1).sum()
ratio = round(neg / pos)
print(f"\nClass imbalance ratio: {ratio}:1")

print("\nSplitting data (time-aware, no shuffle)...")
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, shuffle=False
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Sepsis in test: {yc_test.sum():,} ({yc_test.mean()*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# ADD THESE 3 LINES HERE:
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=yc_train)
print(f"Sample weights calculated. Unique weights: {len(set(sample_weights))}")

# ══════════════════════════════════════════════════════
# TASK 1: CLASSIFICATION
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("TASK 1: CLASSIFICATION - Sepsis Prediction")
print("="*55)

classifiers = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',    # same data, just weighted
        min_samples_leaf=10,        # prevents memorizing majority class
        max_features='sqrt',        # standard best practice
        max_depth=20               # prevents overfitting
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=49,
        max_depth=6,
        learning_rate=0.1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        random_state=42,
        subsample=0.8,
        min_samples_leaf=20,
        max_depth=5
    ),
}

best_clf_score = 0
best_clf_name  = ''
best_clf_model = None
clf_results    = {}

# Replace the best model selection logic with this:
for name, model in classifiers.items():
    print(f"\n  Training {name}...")
    
    if name == 'Gradient Boosting':
        model.fit(X_train_scaled, yc_train, sample_weight=sample_weights)
    else:
        model.fit(X_train_scaled, yc_train)
    
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    auroc   = roc_auc_score(yc_test, proba)
    f1      = f1_score(yc_test, preds, zero_division=0)
    acc     = accuracy_score(yc_test, preds)
    
    # Get sepsis-specific recall from classification report
    from sklearn.metrics import recall_score
    sepsis_recall = recall_score(yc_test, preds, zero_division=0)

    # Combined medical score:
    # 60% weight on recall (catching real sepsis)
    # 40% weight on AUROC (overall discrimination)
    medical_score = (0.6 * sepsis_recall) + (0.4 * auroc)

    print(f"    AUROC={auroc:.4f} | Sepsis Recall={sepsis_recall:.4f} | Medical Score={medical_score:.4f}")
    print(f"\n  --- {name} Classification Report ---")
    print(classification_report(yc_test, preds,
                                target_names=['No Sepsis', 'Sepsis'],
                                zero_division=0))

    clf_results[name] = {
        'AUROC': round(auroc, 4),
        'F1': round(f1, 4),
        'Accuracy': round(acc, 4),
        'Sepsis_Recall': round(sepsis_recall, 4),
        'Medical_Score': round(medical_score, 4)
    }

    # NOW pick best based on medical score not just AUROC
    if medical_score > best_clf_score:
        best_clf_score = medical_score
        best_clf_name  = name
        best_clf_model = model

print(f"  Best Classifier : {best_clf_name} | Medical Score={best_clf_score:.4f}")
joblib.dump(best_clf_model, 'models/best_classifier.pkl')

# ══════════════════════════════════════════════════════
# TASK 2: REGRESSION - Predict NEXT HOUR HR
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("TASK 2: REGRESSION - Next Hour HR Prediction")
print("="*55)

regressors = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest':    RandomForestRegressor(
                            n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':          XGBRegressor(
                            n_estimators=100, random_state=42, verbosity=0),
}

best_reg_score = float('inf')
best_reg_name  = ''
best_reg_model = None
reg_results    = {}

for name, model in regressors.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, yr_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(yr_test, preds))
    mae  = mean_absolute_error(yr_test, preds)
    r2   = r2_score(yr_test, preds)

    reg_results[name] = {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4)
    }
    print(f"    RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

    if rmse < best_reg_score:
        best_reg_score = rmse
        best_reg_name  = name
        best_reg_model = model

print(f"\n  Best Regressor: {best_reg_name} (RMSE={best_reg_score:.4f})")
joblib.dump(best_reg_model, 'models/best_regressor.pkl')
# ══════════════════════════════════════════════════════
# TASK 3: CLUSTERING - Patient Risk Groups
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("TASK 3: CLUSTERING - Patient Risk Groups")
print("="*55)

# Use a sample for training (faster)
sample_size = 50000
sample_X = X_train_scaled[:sample_size]

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(sample_X)
print(f"KMeans trained on {sample_size:,} patient hours")

# ── Get cluster assignments for ALL training data ──────
all_labels = kmeans.predict(X_train_scaled)

# ── Add labels back to original dataframe ─────────────
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
X_train_df['cluster']     = all_labels
X_train_df['sepsislabel'] = yc_train.values

# ── Analyze each cluster's actual vital signs ──────────
print("\nAnalyzing cluster characteristics from actual data...")
print("-"*65)

cluster_stats = X_train_df.groupby('cluster').agg(
    avg_hr         = ('hr',          'mean'),
    avg_temp       = ('temp',        'mean'),
    avg_o2sat      = ('o2sat',       'mean'),
    avg_map        = ('map',         'mean'),
    avg_resp       = ('resp',        'mean'),
    avg_qsofa      = ('qsofa',       'mean'),
    tachy_rate     = ('flag_tachy',  'mean'),
    low_bp_rate    = ('flag_low_bp', 'mean'),
    sepsis_rate    = ('sepsislabel', 'mean'),
    patient_count  = ('hr',          'count')
).round(3)

print(cluster_stats.to_string())
print("-"*65)

# ── Automatically assign risk labels based on sepsis rate ──
# The cluster with highest sepsis rate = most dangerous
# We sort clusters by sepsis rate and assign labels accordingly

sepsis_rates = cluster_stats['sepsis_rate'].to_dict()
sorted_clusters = sorted(sepsis_rates, key=sepsis_rates.get)

# sorted_clusters[0] = lowest sepsis rate = safest
# sorted_clusters[3] = highest sepsis rate = most dangerous
risk_labels = {
    sorted_clusters[0]: 'Stable Low Risk',
    sorted_clusters[1]: 'Elevated Risk',
    sorted_clusters[2]: 'High Risk - Sepsis Likely',
    sorted_clusters[3]: 'Critical - Immediate Attention'
}

print("\nCluster Risk Labels (assigned by actual sepsis rate):")
print("-"*65)
for cluster_id, label in sorted(risk_labels.items()):
    stats      = cluster_stats.loc[cluster_id]
    count      = int(stats['patient_count'])
    sep_rate   = stats['sepsis_rate'] * 100
    avg_hr     = stats['avg_hr']
    avg_o2sat  = stats['avg_o2sat']
    avg_map    = stats['avg_map']

    print(f"\n  Cluster {cluster_id}: {label}")
    print(f"    Patient counts : {count:,}")
    print(f"    Sepsis rate   : {sep_rate:.1f}%")
    print(f"    Avg HR        : {avg_hr:.1f} bpm")
    print(f"    Avg O2Sat     : {avg_o2sat:.1f}%")
    print(f"    Avg MAP       : {avg_map:.1f} mmHg")

# ── Save model and labels ──────────────────────────────
joblib.dump(kmeans,      'models/kmeans_clustering.pkl')
joblib.dump(risk_labels, 'models/cluster_risk_labels.pkl')

cluster_summary = {
    str(k): {
        'risk_label':    v,
        'sepsis_rate':   round(float(sepsis_rates[k]), 4),
        'patient_hours': int(cluster_stats.loc[k, 'patient_count']),
        'avg_hr':        round(float(cluster_stats.loc[k, 'avg_hr']), 1),
        'avg_o2sat':     round(float(cluster_stats.loc[k, 'avg_o2sat']), 1),
        'avg_map':       round(float(cluster_stats.loc[k, 'avg_map']), 1),
    }
    for k, v in risk_labels.items()
}

with open('models/cluster_analysis.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2)

print("\n  Saved -> models/kmeans_clustering.pkl")
print("  Saved -> models/cluster_risk_labels.pkl")
print("  Saved -> models/cluster_analysis.json")
# ══════════════════════════════════════════════════════
# TASK 4: DIMENSIONALITY REDUCTION - PCA
# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("TASK 4: DIMENSIONALITY REDUCTION - PCA")
print("="*55)

# ── Train PCA with 2 components ────────────────────────
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train_scaled)
explained = pca.explained_variance_ratio_

print(f"\n  Variance explained:")
print(f"  PC1 : {explained[0]*100:.1f}% (cardiovascular stress axis)")
print(f"  PC2 : {explained[1]*100:.1f}% (respiratory/temp stress axis)")
print(f"  Total: {sum(explained)*100:.1f}% of 28 features compressed to 2")

# ── What features drive each component ────────────────
components_df = pd.DataFrame(
    pca.components_,
    columns=FEATURE_COLS,
    index=['PC1', 'PC2']
)

pc1_top = components_df.loc['PC1'].abs().nlargest(5)
pc2_top = components_df.loc['PC2'].abs().nlargest(5)

print(f"\n  Top features driving PC1:")
for feat, val in pc1_top.items():
    direction = "+" if components_df.loc['PC1', feat] > 0 else "-"
    print(f"    {direction} {feat}: {val:.3f}")

print(f"\n  Top features driving PC2:")
for feat, val in pc2_top.items():
    direction = "+" if components_df.loc['PC2', feat] > 0 else "-"
    print(f"    {direction} {feat}: {val:.3f}")

# ── Compare sepsis vs non-sepsis in 2D space ──────────
X_train_2d = pca.transform(X_train_scaled)
pca_df     = pd.DataFrame(X_train_2d, columns=['PC1', 'PC2'])
pca_df['sepsis'] = yc_train.values

sepsis_pts    = pca_df[pca_df['sepsis'] == 1]
no_sepsis_pts = pca_df[pca_df['sepsis'] == 0]

pc1_sep = abs(sepsis_pts['PC1'].mean() - no_sepsis_pts['PC1'].mean())
pc2_sep = abs(sepsis_pts['PC2'].mean() - no_sepsis_pts['PC2'].mean())

print(f"\n  Patient positions in 2D health space:")
print(f"  Non-Sepsis: PC1={no_sepsis_pts['PC1'].mean():.3f} | "
      f"PC2={no_sepsis_pts['PC2'].mean():.3f}")
print(f"  Sepsis    : PC1={sepsis_pts['PC1'].mean():.3f} | "
      f"PC2={sepsis_pts['PC2'].mean():.3f}")
print(f"  Group separation: PC1={pc1_sep:.3f} | PC2={pc2_sep:.3f}")

if pc1_sep > 0.3 or pc2_sep > 0.3:
    print(f"  PCA successfully separates sepsis from non-sepsis")
else:
    print(f"  Partial separation - expected for complex clinical data")

# ── How many components needed for higher variance ─────
pca_full   = PCA(random_state=42)
pca_full.fit(X_train_scaled)
cumulative = pca_full.explained_variance_ratio_.cumsum()
n_for_80   = int((cumulative < 0.80).sum() + 1)
n_for_95   = int((cumulative < 0.95).sum() + 1)

print(f"\n  Components needed for 80% variance : {n_for_80}")
print(f"  Components needed for 95% variance : {n_for_95}")
print(f"  We use 2 components for human visualization")

# ── Save models and results ────────────────────────────
joblib.dump(pca, 'models/pca_reducer.pkl')

pca_results = {
    'n_components': 2,
    'variance_explained': {
        'PC1':   round(float(explained[0]), 4),
        'PC2':   round(float(explained[1]), 4),
        'total': round(float(sum(explained)), 4)
    },
    'components_needed_for_80pct': n_for_80,
    'components_needed_for_95pct': n_for_95,
    'pc1_top_features': pc1_top.index.tolist(),
    'pc2_top_features': pc2_top.index.tolist(),
    'group_separation': {
        'pc1': round(float(pc1_sep), 4),
        'pc2': round(float(pc2_sep), 4)
    }
}

with open('models/pca_results.json', 'w') as f:
    json.dump(pca_results, f, indent=2)

print(f"\n  Saved -> models/pca_reducer.pkl")
print(f"  Saved -> models/pca_results.json")

# ══════════════════════════════════════════════════════
# SAVE MANIFEST
# ══════════════════════════════════════════════════════
manifest = {
    'trained_at':            datetime.now().isoformat(),
    'train_rows':            len(X_train),
    'test_rows':             len(X_test),
    'class_imbalance_ratio': ratio,
    'feature_columns':       FEATURE_COLS,
    'classification': {
        'best_model':        best_clf_name,
        'best_medical_score': round(best_clf_score, 4),
        'selection_criteria': '60% sepsis recall + 40% AUROC',
        'all_results':       clf_results
    },
    'regression': {
        'target':      'next_hour_hr',
        'best_model':  best_reg_name,
        'best_rmse':   round(best_reg_score, 4),
        'all_results': reg_results
    },
    'clustering': {
        'algorithm':  'KMeans',
        'n_clusters':  4,
        'risk_labels': {
            str(k): v for k, v in risk_labels.items()
        },
        'cluster_sepsis_rates': {
            str(k): round(float(v), 4)
            for k, v in sepsis_rates.items()
        }
    },
    'dimensionality_reduction': {
        'algorithm':    'PCA',
        'n_components':  2,
        'variance_explained': {
            'PC1':   round(float(explained[0]), 4),
            'PC2':   round(float(explained[1]), 4),
            'total': round(float(sum(explained)), 4)
        },
        'components_for_80pct': n_for_80,
        'components_for_95pct': n_for_95,
        'pc1_top_features':     pc1_top.index.tolist(),
        'pc2_top_features':     pc2_top.index.tolist()
    },
    'time_series': {
        'saved_separately': 'models/timeseries_results.json'
    },
    'recommendation': {
        'saved_separately': 'models/recommendation_results.json'
    },
    'association_rules': {
        'saved_separately': 'models/association_rules.json'
    }
}

with open('models/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

# ══════════════════════════════════════════════════════
print("\n" + "="*55)
print("ALL MODELS TRAINED AND SAVED!")
print("="*55)
print(f"  Best Classifier : {best_clf_name}")
print(f"  Medical Score   : {best_clf_score:.4f}")
print(f"  Best Regressor  : {best_reg_name}")
print(f"  RMSE            : {best_reg_score:.4f}")
print(f"  Clustering      : KMeans 4 clusters")
print(f"  PCA             : {sum(explained)*100:.1f}% variance in 2D")
print(f"  manifest.json   : saved to models/")
print("="*55)

print("\n  Files saved:")
print("  models/best_classifier.pkl")
print("  models/best_regressor.pkl")
print("  models/kmeans_clustering.pkl")
print("  models/cluster_risk_labels.pkl")
print("  models/cluster_analysis.json")
print("  models/pca_reducer.pkl")
print("  models/pca_results.json")
print("  models/scaler.pkl")
print("  models/manifest.json")
print("="*55)
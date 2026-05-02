# # ══════════════════════════════════════════════════════
# # VitalWatch — Prefect ML Orchestration Pipeline
# # File: prefect/flows.py
# # ══════════════════════════════════════════════════════
# # HOW TO RUN:
# #   Step 1 (one time):  pip install prefect
# #   Step 2 (terminal1): prefect server start
# #   Step 3 (terminal2): cd C:\Users\DELL\Desktop\VitalWatch
# #                        python prefect/flows.py
# #   Step 4: Open http://localhost:4200 to see the UI
# # ══════════════════════════════════════════════════════

# import sys
# import os

# # ── Make sure src/ is importable ──────────────────────
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, ROOT)
# sys.path.insert(0, os.path.join(ROOT, 'src'))

# import pandas as pd
# import numpy as np
# import joblib
# import json
# from datetime import datetime, timedelta

# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import (RandomForestClassifier,
#                                GradientBoostingClassifier,
#                                RandomForestRegressor)
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
#                              mean_squared_error, mean_absolute_error,
#                              r2_score, recall_score, classification_report)
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.utils.class_weight import compute_sample_weight
# from xgboost import XGBClassifier, XGBRegressor

# from prefect import flow, task, get_run_logger
# from prefect.tasks import task_input_hash

# # ── Config ─────────────────────────────────────────────
# DB_URL     = 'postgresql://postgres:1234@localhost:5432/vitalwatch_db'
# MODELS_DIR = os.path.join(ROOT, 'models')
# os.makedirs(MODELS_DIR, exist_ok=True)

# FEATURE_COLS = [
#     'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
#     'age', 'gender',
#     'hr_mean_3h', 'hr_mean_6h', 'sbp_mean_3h', 'map_mean_3h', 'resp_mean_3h',
#     'hr_trend', 'sbp_trend', 'temp_trend', 'resp_trend',
#     'hr_std_3h', 'sbp_std_3h',
#     'flag_tachy', 'flag_hypoxia', 'flag_fever', 'flag_hypotemp',
#     'flag_low_bp', 'flag_tachypnea', 'qsofa', 'hours_in_icu'
# ]

# # ── Notification (uncomment when Discord webhook is ready) ──
# # DISCORD_WEBHOOK = "https://discord.com/api/webhooks/YOUR_WEBHOOK_HERE"
# # async def notify(msg):
# #     from prefect.blocks.notifications import DiscordWebhook
# #     await DiscordWebhook(url=DISCORD_WEBHOOK).notify(msg)


# # ══════════════════════════════════════════════════════
# # TASK 1 — FEATURE ENGINEERING (calls your src/feature_engineering.py logic)
# # ══════════════════════════════════════════════════════
# @task(
#     name="Feature Engineering",
#     description="Load raw vitals from patient_vitals, engineer features, save to patient_features",
#     retries=3,
#     retry_delay_seconds=10,
# )
# def run_feature_engineering() -> bool:
#     logger = get_run_logger()
#     logger.info("Starting feature engineering from src/feature_engineering.py logic...")

#     try:
#         engine = create_engine(DB_URL)

#         logger.info("Loading patient_vitals from PostgreSQL...")
#         df = pd.read_sql("SELECT * FROM patient_vitals", engine)
#         df.columns = [col.strip().lower() for col in df.columns]
#         logger.info(f"Loaded {len(df):,} rows | {df['patient_id'].nunique():,} unique patients")

#         all_patients = []
#         total = df['patient_id'].nunique()

#         for i, (patient_id, group) in enumerate(df.groupby('patient_id')):
#             g = group.copy().sort_values('iculos')

#             # Rolling averages
#             g['hr_mean_3h']   = g['hr'].rolling(window=3,  min_periods=1).mean()
#             g['hr_mean_6h']   = g['hr'].rolling(window=6,  min_periods=1).mean()
#             g['sbp_mean_3h']  = g['sbp'].rolling(window=3, min_periods=1).mean()
#             g['map_mean_3h']  = g['map'].rolling(window=3, min_periods=1).mean()
#             g['resp_mean_3h'] = g['resp'].rolling(window=3, min_periods=1).mean()

#             # Trends (rate of change)
#             g['hr_trend']   = g['hr'].diff()
#             g['sbp_trend']  = g['sbp'].diff()
#             g['temp_trend'] = g['temp'].diff()
#             g['resp_trend'] = g['resp'].diff()

#             # Volatility
#             g['hr_std_3h']  = g['hr'].rolling(window=3,  min_periods=1).std().fillna(0)
#             g['sbp_std_3h'] = g['sbp'].rolling(window=3, min_periods=1).std().fillna(0)

#             # Clinical alarm flags
#             g['flag_tachy']     = (g['hr']    > 100).astype(int)
#             g['flag_hypoxia']   = (g['o2sat'] < 92).astype(int)
#             g['flag_fever']     = (g['temp']  > 38.3).astype(int)
#             g['flag_hypotemp']  = (g['temp']  < 36.0).astype(int)
#             g['flag_low_bp']    = (g['map']   < 65).astype(int)
#             g['flag_tachypnea'] = (g['resp']  > 22).astype(int)

#             # qSOFA score
#             g['qsofa']        = g['flag_tachypnea'] + g['flag_low_bp']
#             g['hours_in_icu'] = g['iculos']

#             all_patients.append(g)

#             if (i + 1) % 2000 == 0:
#                 logger.info(f"  Processed {i+1:,} / {total:,} patients...")

#         logger.info("Combining all patient records...")
#         df_features = pd.concat(all_patients, ignore_index=True).fillna(0)
#         logger.info(f"Final feature table shape: {df_features.shape}")

#         logger.info("Saving patient_features to PostgreSQL...")
#         df_features.to_sql('patient_features', engine, if_exists='replace', index=False)
#         logger.info("Feature engineering complete — patient_features table updated")
#         return True

#     except Exception as e:
#         logger.error(f"Feature engineering failed: {e}")
#         raise


# # ══════════════════════════════════════════════════════
# # TASK 2 — DATA INGESTION
# # ══════════════════════════════════════════════════════
# @task(
#     name="Data Ingestion",
#     description="Load engineered features from patient_features table",
#     retries=3,
#     retry_delay_seconds=10,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(hours=1),
# )
# def ingest_data() -> pd.DataFrame:
#     logger = get_run_logger()
#     logger.info("Loading patient_features from PostgreSQL...")

#     try:
#         engine = create_engine(DB_URL)
#         df = pd.read_sql("SELECT * FROM patient_features", engine)
#         logger.info(f"Loaded {len(df):,} rows | {df.shape[1]} columns")

#         if df.empty:
#             raise ValueError("patient_features table is empty!")

#         missing = [c for c in FEATURE_COLS if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing feature columns: {missing}")

#         logger.info("Data validation passed")
#         return df

#     except Exception as e:
#         logger.error(f"Data ingestion failed: {e}")
#         raise


# # ══════════════════════════════════════════════════════
# # TASK 3 — DATA SPLITTING & SCALING
# # ══════════════════════════════════════════════════════
# @task(
#     name="Prepare Training Data",
#     description="Engineer target, split train/test, scale features",
#     retries=2,
#     retry_delay_seconds=5,
# )
# def prepare_data(df: pd.DataFrame) -> dict:
#     logger = get_run_logger()
#     logger.info("Preparing training data...")

#     # Target: next-hour HR
#     df['hr_next_hour'] = df.groupby('patient_id')['hr'].shift(-1)
#     df = df.dropna(subset=['hr_next_hour'])

#     X       = df[FEATURE_COLS].fillna(0)
#     y_class = df['sepsislabel']
#     y_reg   = df['hr_next_hour']

#     neg, pos = (y_class == 0).sum(), (y_class == 1).sum()
#     ratio    = round(neg / pos)
#     logger.info(f"Class ratio: {ratio}:1 | Sepsis cases: {pos:,} ({pos/(pos+neg)*100:.1f}%)")

#     # Time-aware split — no shuffle to preserve temporal order
#     X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
#         X, y_class, y_reg, test_size=0.2, shuffle=False
#     )
#     logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

#     scaler         = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled  = scaler.transform(X_test)
#     joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
#     logger.info("Scaler saved -> models/scaler.pkl")

#     sample_weights = compute_sample_weight(class_weight='balanced', y=yc_train)

#     return {
#         'X_train':        X_train,
#         'X_test':         X_test,
#         'X_train_scaled': X_train_scaled,
#         'X_test_scaled':  X_test_scaled,
#         'yc_train':       yc_train,
#         'yc_test':        yc_test,
#         'yr_train':       yr_train,
#         'yr_test':        yr_test,
#         'sample_weights': sample_weights,
#         'ratio':          ratio,
#     }


# # ══════════════════════════════════════════════════════
# # TASK 4 — CLASSIFICATION (from src/train_models.py)
# # ══════════════════════════════════════════════════════
# @task(
#     name="Train Classifiers",
#     description="Train 4 classifiers, pick best by medical score, save",
#     retries=1,
#     retry_delay_seconds=5,
# )
# def train_classifiers(data: dict) -> dict:
#     logger = get_run_logger()
#     logger.info("Training classifiers (Logistic, RF, XGBoost, GBM)...")

#     Xtr, Xte = data['X_train_scaled'], data['X_test_scaled']
#     yc_tr, yc_te = data['yc_train'], data['yc_test']
#     sw, ratio = data['sample_weights'], data['ratio']

#     classifiers = {
#         'Logistic Regression': LogisticRegression(
#             max_iter=1000, random_state=42, class_weight='balanced'),
#         'Random Forest': RandomForestClassifier(
#             n_estimators=100, random_state=42, n_jobs=-1,
#             class_weight='balanced', min_samples_leaf=10,
#             max_features='sqrt', max_depth=20),
#         'XGBoost': XGBClassifier(
#             n_estimators=200, random_state=42, eval_metric='logloss',
#             verbosity=0, scale_pos_weight=ratio, max_depth=6, learning_rate=0.1),
#         'Gradient Boosting': GradientBoostingClassifier(
#             n_estimators=200, random_state=42,
#             subsample=0.8, min_samples_leaf=20, max_depth=5),
#     }

#     best_score, best_name, best_model = 0, '', None
#     results = {}

#     for name, model in classifiers.items():
#         logger.info(f"  Training {name}...")
#         try:
#             if name == 'Gradient Boosting':
#                 model.fit(Xtr, yc_tr, sample_weight=sw)
#             else:
#                 model.fit(Xtr, yc_tr)

#             preds  = model.predict(Xte)
#             proba  = model.predict_proba(Xte)[:, 1]
#             auroc  = roc_auc_score(yc_te, proba)
#             f1     = f1_score(yc_te, preds, zero_division=0)
#             acc    = accuracy_score(yc_te, preds)
#             recall = recall_score(yc_te, preds, zero_division=0)
#             med    = (0.6 * recall) + (0.4 * auroc)   # medical score

#             results[name] = {
#                 'AUROC': round(auroc, 4), 'F1': round(f1, 4),
#                 'Accuracy': round(acc, 4), 'Sepsis_Recall': round(recall, 4),
#                 'Medical_Score': round(med, 4)
#             }
#             logger.info(f"  {name}: AUROC={auroc:.4f} | Recall={recall:.4f} | Medical={med:.4f}")

#             if med > best_score:
#                 best_score, best_name, best_model = med, name, model

#         except Exception as e:
#             logger.warning(f"  {name} failed: {e} — skipping")

#     joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_classifier.pkl'))
#     logger.info(f"Best: {best_name} (Medical Score={best_score:.4f}) -> models/best_classifier.pkl")

#     return {'best_name': best_name, 'best_score': round(best_score, 4), 'all_results': results}


# # ══════════════════════════════════════════════════════
# # TASK 5 — REGRESSION (from src/train_models.py)
# # ══════════════════════════════════════════════════════
# @task(
#     name="Train Regressors",
#     description="Train 3 regressors for next-hour HR, pick best RMSE, save",
#     retries=1,
#     retry_delay_seconds=5,
# )
# def train_regressors(data: dict) -> dict:
#     logger = get_run_logger()
#     logger.info("Training regressors (Ridge, RF, XGBoost)...")

#     Xtr, Xte   = data['X_train_scaled'], data['X_test_scaled']
#     yr_tr, yr_te = data['yr_train'], data['yr_test']

#     regressors = {
#         'Ridge Regression': Ridge(alpha=1.0),
#         'Random Forest':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
#         'XGBoost':          XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
#     }

#     best_rmse, best_name, best_model = float('inf'), '', None
#     results = {}

#     for name, model in regressors.items():
#         logger.info(f"  Training {name}...")
#         try:
#             model.fit(Xtr, yr_tr)
#             preds = model.predict(Xte)
#             rmse  = np.sqrt(mean_squared_error(yr_te, preds))
#             mae   = mean_absolute_error(yr_te, preds)
#             r2    = r2_score(yr_te, preds)

#             results[name] = {'RMSE': round(rmse,4), 'MAE': round(mae,4), 'R2': round(r2,4)}
#             logger.info(f"  {name}: RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

#             if rmse < best_rmse:
#                 best_rmse, best_name, best_model = rmse, name, model

#         except Exception as e:
#             logger.warning(f"  {name} failed: {e} — skipping")

#     joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_regressor.pkl'))
#     logger.info(f"Best: {best_name} (RMSE={best_rmse:.4f}) -> models/best_regressor.pkl")

#     return {'best_name': best_name, 'best_rmse': round(best_rmse, 4), 'all_results': results}


# # ══════════════════════════════════════════════════════
# # TASK 6 — CLUSTERING
# # ══════════════════════════════════════════════════════
# @task(
#     name="Train Clustering",
#     description="KMeans 4-cluster patient risk grouping",
#     retries=1,
#     retry_delay_seconds=5,
# )
# def train_clustering(data: dict) -> dict:
#     logger = get_run_logger()
#     logger.info("Training KMeans clustering...")

#     X_train        = data['X_train']
#     X_train_scaled = data['X_train_scaled']
#     yc_train       = data['yc_train']

#     sample_size = min(50000, len(X_train_scaled))
#     kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
#     kmeans.fit(X_train_scaled[:sample_size])
#     logger.info(f"KMeans trained on {sample_size:,} samples")

#     all_labels = kmeans.predict(X_train_scaled)
#     df_tmp = pd.DataFrame(X_train, columns=FEATURE_COLS)
#     df_tmp['cluster']     = all_labels
#     df_tmp['sepsislabel'] = yc_train.values

#     stats = df_tmp.groupby('cluster').agg(
#         avg_hr        = ('hr',          'mean'),
#         avg_o2sat     = ('o2sat',       'mean'),
#         avg_map       = ('map',         'mean'),
#         sepsis_rate   = ('sepsislabel', 'mean'),
#         patient_count = ('hr',          'count')
#     ).round(3)

#     sepsis_rates    = stats['sepsis_rate'].to_dict()
#     sorted_clusters = sorted(sepsis_rates, key=sepsis_rates.get)
#     risk_labels = {
#         sorted_clusters[0]: 'Stable Low Risk',
#         sorted_clusters[1]: 'Elevated Risk',
#         sorted_clusters[2]: 'High Risk - Sepsis Likely',
#         sorted_clusters[3]: 'Critical - Immediate Attention'
#     }

#     for cid, label in sorted(risk_labels.items()):
#         logger.info(f"  Cluster {cid}: {label} | Sepsis: {sepsis_rates[cid]*100:.1f}%")

#     joblib.dump(kmeans,      os.path.join(MODELS_DIR, 'kmeans_clustering.pkl'))
#     joblib.dump(risk_labels, os.path.join(MODELS_DIR, 'cluster_risk_labels.pkl'))

#     summary = {
#         str(k): {
#             'risk_label':    v,
#             'sepsis_rate':   round(float(sepsis_rates[k]), 4),
#             'patient_hours': int(stats.loc[k, 'patient_count']),
#             'avg_hr':        round(float(stats.loc[k, 'avg_hr']), 1),
#             'avg_o2sat':     round(float(stats.loc[k, 'avg_o2sat']), 1),
#             'avg_map':       round(float(stats.loc[k, 'avg_map']), 1),
#         } for k, v in risk_labels.items()
#     }
#     with open(os.path.join(MODELS_DIR, 'cluster_analysis.json'), 'w') as f:
#         json.dump(summary, f, indent=2)

#     logger.info("Saved -> kmeans_clustering.pkl, cluster_risk_labels.pkl, cluster_analysis.json")
#     return {
#         'risk_labels':   {str(k): v for k, v in risk_labels.items()},
#         'sepsis_rates':  {str(k): round(float(v), 4) for k, v in sepsis_rates.items()}
#     }


# # ══════════════════════════════════════════════════════
# # TASK 7 — PCA
# # ══════════════════════════════════════════════════════
# @task(
#     name="Train PCA",
#     description="PCA dimensionality reduction to 2 components",
#     retries=1,
#     retry_delay_seconds=5,
# )
# def train_pca(data: dict) -> dict:
#     logger = get_run_logger()
#     logger.info("Training PCA (2 components)...")

#     X_train_scaled = data['X_train_scaled']
#     yc_train       = data['yc_train']

#     pca       = PCA(n_components=2, random_state=42)
#     pca.fit(X_train_scaled)
#     explained = pca.explained_variance_ratio_
#     logger.info(f"PC1: {explained[0]*100:.1f}% | PC2: {explained[1]*100:.1f}% | Total: {sum(explained)*100:.1f}%")

#     # Sepsis separation in 2D
#     X_2d   = pca.transform(X_train_scaled)
#     pca_df = pd.DataFrame(X_2d, columns=['PC1','PC2'])
#     pca_df['sepsis'] = yc_train.values
#     sep1 = abs(pca_df[pca_df.sepsis==1]['PC1'].mean() - pca_df[pca_df.sepsis==0]['PC1'].mean())
#     sep2 = abs(pca_df[pca_df.sepsis==1]['PC2'].mean() - pca_df[pca_df.sepsis==0]['PC2'].mean())
#     logger.info(f"Group separation — PC1: {sep1:.3f} | PC2: {sep2:.3f}")

#     # Variance thresholds
#     pca_full   = PCA(random_state=42)
#     pca_full.fit(X_train_scaled)
#     cumulative = pca_full.explained_variance_ratio_.cumsum()
#     n80 = int((cumulative < 0.80).sum() + 1)
#     n95 = int((cumulative < 0.95).sum() + 1)
#     logger.info(f"Components for 80% variance: {n80} | For 95%: {n95}")

#     joblib.dump(pca, os.path.join(MODELS_DIR, 'pca_reducer.pkl'))

#     result = {
#         'n_components': 2,
#         'variance_explained': {
#             'PC1': round(float(explained[0]), 4),
#             'PC2': round(float(explained[1]), 4),
#             'total': round(float(sum(explained)), 4)
#         },
#         'components_needed_for_80pct': n80,
#         'components_needed_for_95pct': n95,
#         'group_separation': {'pc1': round(float(sep1),4), 'pc2': round(float(sep2),4)}
#     }
#     with open(os.path.join(MODELS_DIR, 'pca_results.json'), 'w') as f:
#         json.dump(result, f, indent=2)

#     logger.info("Saved -> pca_reducer.pkl, pca_results.json")
#     return result


# # ══════════════════════════════════════════════════════
# # TASK 8 — SAVE MANIFEST
# # ══════════════════════════════════════════════════════
# @task(
#     name="Save Manifest",
#     description="Compile and save manifest.json with all model results",
# )
# def save_manifest(clf: dict, reg: dict, cluster: dict,
#                   pca: dict, data: dict) -> str:
#     logger = get_run_logger()

#     manifest = {
#         'trained_at':            datetime.now().isoformat(),
#         'pipeline':              'prefect/flows.py',
#         'train_rows':            len(data['X_train']),
#         'test_rows':             len(data['X_test']),
#         'class_imbalance_ratio': data['ratio'],
#         'feature_columns':       FEATURE_COLS,
#         'classification': {
#             'best_model':         clf['best_name'],
#             'best_medical_score': clf['best_score'],
#             'selection_criteria': '60% sepsis recall + 40% AUROC',
#             'all_results':        clf['all_results']
#         },
#         'regression': {
#             'target':      'next_hour_hr',
#             'best_model':  reg['best_name'],
#             'best_rmse':   reg['best_rmse'],
#             'all_results': reg['all_results']
#         },
#         'clustering': {
#             'algorithm':           'KMeans',
#             'n_clusters':           4,
#             'risk_labels':          cluster['risk_labels'],
#             'cluster_sepsis_rates': cluster['sepsis_rates']
#         },
#         'dimensionality_reduction': {
#             'algorithm':           'PCA',
#             'n_components':         2,
#             'variance_explained':   pca['variance_explained'],
#             'components_for_80pct': pca['components_needed_for_80pct'],
#             'components_for_95pct': pca['components_needed_for_95pct'],
#         },
#         'time_series':    {'saved_separately': 'models/timeseries_results.json'},
#         'recommendation': {'saved_separately': 'models/recommendation_results.json'},
#         'association_rules': {'saved_separately': 'models/association_rules.json'}
#     }

#     path = os.path.join(MODELS_DIR, 'manifest.json')
#     with open(path, 'w') as f:
#         json.dump(manifest, f, indent=2)

#     logger.info("=" * 55)
#     logger.info("PIPELINE COMPLETE — SUMMARY")
#     logger.info("=" * 55)
#     logger.info(f"Best Classifier : {clf['best_name']} | Medical Score: {clf['best_score']}")
#     logger.info(f"Best Regressor  : {reg['best_name']} | RMSE: {reg['best_rmse']}")
#     logger.info(f"Clustering      : KMeans 4 clusters")
#     logger.info(f"PCA             : {pca['variance_explained']['total']*100:.1f}% variance in 2D")
#     logger.info(f"Manifest saved  : {path}")
#     logger.info("=" * 55)

#     # ── Uncomment when Discord is ready ───────────────
#     # import asyncio
#     # asyncio.run(notify(
#     #     f"✅ VitalWatch pipeline finished!\n"
#     #     f"Classifier: {clf['best_name']} (score={clf['best_score']})\n"
#     #     f"Regressor : {reg['best_name']} (RMSE={reg['best_rmse']})"
#     # ))

#     return path


# # ══════════════════════════════════════════════════════
# # MAIN FLOW
# # ══════════════════════════════════════════════════════
# @flow(
#     name="VitalWatch ML Pipeline",
#     description=(
#         "End-to-end ML pipeline: "
#         "Feature Engineering → Ingest → Prepare → "
#         "Classify → Regress → Cluster → PCA → Save Manifest"
#     ),
# )
# def vitalwatch_pipeline(skip_feature_engineering: bool = False):
#     """
#     Args:
#         skip_feature_engineering: Set True if patient_features table
#                                   already exists and is up to date.
#                                   Saves time on reruns.
#     """
#     logger = get_run_logger()
#     logger.info("VitalWatch ML Pipeline — Starting")

#     # Step 1: Feature engineering (can skip if table already exists)
#     if not skip_feature_engineering:
#         run_feature_engineering()
#     else:
#         logger.info("Skipping feature engineering (skip_feature_engineering=True)")

#     # Step 2: Load engineered features
#     df = ingest_data()

#     # Step 3: Prepare / split / scale
#     data = prepare_data(df)

#     # Steps 4-7: Train all models
#     clf_result     = train_classifiers(data)
#     reg_result     = train_regressors(data)
#     cluster_result = train_clustering(data)
#     pca_result     = train_pca(data)

#     # Step 8: Save manifest
#     manifest_path  = save_manifest(
#         clf_result, reg_result, cluster_result, pca_result, data
#     )

#     logger.info(f"All done! Manifest: {manifest_path}")
#     return manifest_path


# # ══════════════════════════════════════════════════════
# if __name__ == "__main__":
#     # Change skip_feature_engineering=True to skip re-engineering
#     # if your patient_features table is already up to date
#     vitalwatch_pipeline(skip_feature_engineering=False)











from prefect import flow, task
from prefect.logging import get_run_logger
import subprocess
import sys
import os

BASE = os.path.join(os.path.dirname(__file__), '..')

@task(name="Data Validation", retries=2, retry_delay_seconds=10)
def validate_data():
    logger = get_run_logger()
    logger.info("Validating data in PostgreSQL...")
    from sqlalchemy import create_engine
    import pandas as pd
    engine = create_engine(
        'postgresql://postgres:1234@localhost:5432/vitalwatch_db'
    )
    count = pd.read_sql(
        "SELECT COUNT(*) as c FROM patient_features", engine
    ).iloc[0]['c']
    assert count > 700000, f"Not enough rows: {count}"
    logger.info(f"Data valid: {count:,} rows found")
    return count

@task(name="Feature Engineering", retries=1)
def run_feature_engineering():
    logger = get_run_logger()
    logger.info("Running feature engineering...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'feature_engineering.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Feature engineering failed: {result.stderr}")
    logger.info("Feature engineering complete")
    return True

@task(name="Model Training", retries=1)
def run_model_training():
    logger = get_run_logger()
    logger.info("Training all ML models...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'train_models.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
    logger.info("Model training complete")
    return True

@task(name="Time Series Training", retries=1)
def run_timeseries():
    logger = get_run_logger()
    logger.info("Running time series training...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'time_series.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Time series failed: {result.stderr}")
    logger.info("Time series complete")
    return True

@task(name="Recommendation System", retries=1)
def run_recommendation():
    logger = get_run_logger()
    logger.info("Running recommendation system...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'recomendations.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Recommendation failed: {result.stderr}")
    logger.info("Recommendation system complete")
    return True

@task(name="Association Rules", retries=1)
def run_association_rules():
    logger = get_run_logger()
    logger.info("Running association rules...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'src', 'association_rules.py')],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Association rules failed: {result.stderr}")
    logger.info("Association rules complete")
    return True

@task(name="ML Tests", retries=1)
def run_tests():
    logger = get_run_logger()
    logger.info("Running ML pipeline tests...")
    result = subprocess.run(
        [sys.executable,
         os.path.join(BASE, 'tests', 'test_pipeline.py')],
        capture_output=True, text=True
    )
    logger.info(result.stdout[-500:])
    if result.returncode != 0:
        raise Exception(f"Tests failed: {result.stderr}")
    logger.info("All tests passed")
    return True

@task(name="Send Notification")
def send_notification(success: bool, message: str):
    logger = get_run_logger()
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Pipeline {status}: {message}")
    # Discord webhook would go here
    # For now just logs
    print(f"\nVitalWatch Pipeline {status}")
    print(f"Message: {message}")

# ══════════════════════════════════════════════
# MAIN PIPELINE FLOW
# ══════════════════════════════════════════════
@flow(name="VitalWatch ML Pipeline",
      description="Full ML training pipeline for sepsis prediction")
def vitalwatch_pipeline(
    run_training: bool = True,
    run_feature_eng: bool = True
):
    logger = get_run_logger()
    logger.info("Starting VitalWatch ML Pipeline...")

    try:
        # Step 1: Validate data
        row_count = validate_data()

        # Step 2: Feature engineering (optional)
        if run_feature_eng:
            run_feature_engineering()

        # Step 3: Train all models (parallel where possible)
        if run_training:
            run_model_training()
            run_timeseries()
            run_recommendation()
            run_association_rules()

        # Step 4: Run tests
        run_tests()

        # Step 5: Notify success
        send_notification(
            True,
            f"All models trained successfully on {row_count:,} rows"
        )
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        send_notification(False, str(e))
        raise

# ── Quick test flow (runs tests only, no retraining) ──
@flow(name="VitalWatch Test Only")
def test_only_flow():
    validate_data()
    run_tests()
    send_notification(True, "Tests passed — no retraining")

if __name__ == "__main__":
    vitalwatch_pipeline()
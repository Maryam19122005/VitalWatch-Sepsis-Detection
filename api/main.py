# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# import joblib
# import json
# import numpy as np
# import pandas as pd
# import os

# # ── Initialize FastAPI app ─────────────────────────────
# app = FastAPI(
#     title="VitalWatch API",
#     description="Sepsis Early Warning & ICU Patient Deterioration Prediction System",
#     version="1.0.0"
# )

# # ── Feature columns (must match training order) ────────
# FEATURE_COLS = [
#     'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
#     'age', 'gender',
#     'hr_mean_3h', 'hr_mean_6h', 'sbp_mean_3h',
#     'map_mean_3h', 'resp_mean_3h',
#     'hr_trend', 'sbp_trend', 'temp_trend', 'resp_trend',
#     'hr_std_3h', 'sbp_std_3h',
#     'flag_tachy', 'flag_hypoxia', 'flag_fever',
#     'flag_hypotemp', 'flag_low_bp', 'flag_tachypnea',
#     'qsofa', 'hours_in_icu'
# ]

# # ── Load all models at startup ─────────────────────────
# print("Loading all VitalWatch models...")

# BASE = os.path.join(os.path.dirname(__file__), '..', 'models')

# classifier         = joblib.load(os.path.join(BASE, 'best_classifier.pkl'))
# regressor          = joblib.load(os.path.join(BASE, 'best_regressor.pkl'))
# scaler             = joblib.load(os.path.join(BASE, 'scaler.pkl'))
# kmeans             = joblib.load(os.path.join(BASE, 'kmeans_clustering.pkl'))
# cluster_labels     = joblib.load(os.path.join(BASE, 'cluster_risk_labels.pkl'))
# pca                = joblib.load(os.path.join(BASE, 'pca_reducer.pkl'))
# knn                = joblib.load(os.path.join(BASE, 'knn_recommendation.pkl'))
# rec_scaler         = joblib.load(os.path.join(BASE, 'recommendation_scaler.pkl'))
# patient_profiles   = joblib.load(os.path.join(BASE, 'patient_profiles.pkl'))
# timeseries_model   = joblib.load(os.path.join(BASE, 'best_timeseries.pkl'))

# with open(os.path.join(BASE, 'manifest.json')) as f:
#     manifest = json.load(f)

# with open(os.path.join(BASE, 'association_rules.json')) as f:
#     assoc_rules = json.load(f)

# print("All models loaded successfully!")

# # ══════════════════════════════════════════════════════
# # INPUT SCHEMAS
# # ══════════════════════════════════════════════════════

# class PatientVitals(BaseModel):
#     """Full patient vitals for sepsis prediction"""
#     hr:           float = 85.0
#     o2sat:        float = 96.0
#     temp:         float = 37.0
#     sbp:          float = 120.0
#     map:          float = 80.0
#     dbp:          float = 70.0
#     resp:         float = 16.0
#     age:          float = 50.0
#     gender:       float = 1.0
#     hr_mean_3h:   float = 85.0
#     hr_mean_6h:   float = 85.0
#     sbp_mean_3h:  float = 120.0
#     map_mean_3h:  float = 80.0
#     resp_mean_3h: float = 16.0
#     hr_trend:     float = 0.0
#     sbp_trend:    float = 0.0
#     temp_trend:   float = 0.0
#     resp_trend:   float = 0.0
#     hr_std_3h:    float = 2.0
#     sbp_std_3h:   float = 3.0
#     flag_tachy:   float = 0.0
#     flag_hypoxia: float = 0.0
#     flag_fever:   float = 0.0
#     flag_hypotemp:float = 0.0
#     flag_low_bp:  float = 0.0
#     flag_tachypnea:float = 0.0
#     qsofa:        float = 0.0
#     hours_in_icu: float = 1.0

# class TimeSeriesInput(BaseModel):
#     """Last 6 hours of HR for forecasting"""
#     hr_last_6_hours: list[float]
#     patient_id: Optional[str] = "unknown"

# class RecommendationInput(BaseModel):
#     """Patient profile for similar case finding"""
#     avg_hr:          float = 85.0
#     avg_temp:        float = 37.0
#     avg_o2sat:       float = 96.0
#     avg_sbp:         float = 120.0
#     avg_map:         float = 80.0
#     avg_resp:        float = 16.0
#     max_qsofa:       float = 0.0
#     had_tachy:       float = 0.0
#     had_fever:       float = 0.0
#     had_low_bp:      float = 0.0
#     had_hypoxia:     float = 0.0
#     total_icu_hours: float = 24.0

# # ══════════════════════════════════════════════════════
# # HELPER FUNCTIONS
# # ══════════════════════════════════════════════════════

# def prepare_features(vitals: PatientVitals) -> np.ndarray:
#     """Convert patient vitals to scaled feature array"""
#     values = np.array([[getattr(vitals, col) for col in FEATURE_COLS]])
#     return scaler.transform(values)

# def get_risk_level(probability: float) -> str:
#     """Convert sepsis probability to risk level"""
#     if probability >= 0.7:
#         return "HIGH RISK"
#     elif probability >= 0.4:
#         return "MODERATE RISK"
#     else:
#         return "LOW RISK"

# def get_clinical_action(risk_level: str) -> str:
#     """Get clinical action based on risk level"""
#     actions = {
#         "HIGH RISK":     "Initiate sepsis protocol immediately. "
#                          "Order blood cultures and lactate. "
#                          "Start IV antibiotics within 1 hour.",
#         "MODERATE RISK": "Increase monitoring to every 30 minutes. "
#                          "Order CBC and metabolic panel. "
#                          "Notify attending physician.",
#         "LOW RISK":      "Continue standard monitoring every 4 hours. "
#                          "No immediate intervention required."
#     }
#     return actions.get(risk_level, "Follow standard protocol")

# # ══════════════════════════════════════════════════════
# # ENDPOINTS
# # ══════════════════════════════════════════════════════

# # ── 1. Health Check ────────────────────────────────────
# @app.get("/")
# def root():
#     return {
#         "system":  "VitalWatch",
#         "status":  "running",
#         "version": "1.0.0",
#         "message": "Sepsis Early Warning System is active"
#     }

# # ── 2. Model Info ──────────────────────────────────────
# @app.get("/model-info")
# def model_info():
#     return {
#         "trained_at":       manifest.get("trained_at"),
#         "train_rows":       manifest.get("train_rows"),
#         "best_classifier":  manifest["classification"]["best_model"],
#         "medical_score":    manifest["classification"]["best_medical_score"],
#         "best_regressor":   manifest["regression"]["best_model"],
#         "regression_rmse":  manifest["regression"]["best_rmse"],
#         "all_clf_results":  manifest["classification"]["all_results"],
#         "all_reg_results":  manifest["regression"]["all_results"]
#     }

# # ── 3. Feature Columns ─────────────────────────────────
# @app.get("/feature-columns")
# def feature_columns():
#     return {
#         "total_features": len(FEATURE_COLS),
#         "columns":        FEATURE_COLS
#     }

# # ── 4. Sepsis Prediction (CLASSIFICATION) ─────────────
# @app.post("/predict/sepsis")
# def predict_sepsis(patient: PatientVitals):
#     try:
#         # Prepare features
#         X_scaled = prepare_features(patient)

#         # Get prediction and probability
#         prediction  = int(classifier.predict(X_scaled)[0])
#         probability = float(classifier.predict_proba(X_scaled)[0][1])
#         risk_level  = get_risk_level(probability)
#         action      = get_clinical_action(risk_level)

#         # Clinical flags
#         flags = []
#         if patient.flag_tachy:    flags.append("Tachycardia")
#         if patient.flag_fever:    flags.append("Fever")
#         if patient.flag_hypoxia:  flags.append("Hypoxia")
#         if patient.flag_low_bp:   flags.append("Low Blood Pressure")
#         if patient.flag_tachypnea:flags.append("Tachypnea")
#         if patient.qsofa >= 2:    flags.append("High qSOFA Score")

#         return {
#             "prediction":       prediction,
#             "sepsis_risk":      prediction == 1,
#             "probability":      round(probability, 4),
#             "risk_level":       risk_level,
#             "clinical_flags":   flags,
#             "recommended_action": action,
#             "model_used":       manifest["classification"]["best_model"]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 5. Next Hour HR Prediction (REGRESSION) ───────────
# @app.post("/predict/hr-next-hour")
# def predict_hr(patient: PatientVitals):
#     try:
#         X_scaled    = prepare_features(patient)
#         predicted_hr = float(regressor.predict(X_scaled)[0])

#         # Clinical interpretation
#         if predicted_hr > 120:
#             hr_status = "CRITICAL - Severe tachycardia predicted"
#         elif predicted_hr > 100:
#             hr_status = "WARNING - Tachycardia threshold will be crossed"
#         elif predicted_hr < 60:
#             hr_status = "WARNING - Bradycardia predicted"
#         else:
#             hr_status = "NORMAL - HR within acceptable range"

#         trend = predicted_hr - patient.hr
#         trend_direction = "rising" if trend > 2 else \
#                          "falling" if trend < -2 else "stable"

#         return {
#             "current_hr":       patient.hr,
#             "predicted_next_hr": round(predicted_hr, 1),
#             "change":           round(trend, 1),
#             "trend_direction":  trend_direction,
#             "hr_status":        hr_status,
#             "model_used":       manifest["regression"]["best_model"]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 6. Vital Sign Forecasting (TIME SERIES) ───────────
# @app.post("/forecast/vitals")
# def forecast_vitals(data: TimeSeriesInput):
#     try:
#         hr_values = data.hr_last_6_hours

#         # Validate input
#         if len(hr_values) != 6:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Need exactly 6 hours of HR. Got {len(hr_values)}"
#             )

#         window = np.array(hr_values)

#         # Add engineered features (same as training)
#         mean  = window.mean()
#         std   = window.std()
#         trend = window[-1] - window[0]

#         features = np.array([[*window, mean, std, trend]])
#         predicted = float(timeseries_model.predict(features)[0])

#         # Generate 6-hour forecast by rolling predictions
#         forecast = []
#         current_window = list(window)

#         for hour in range(1, 7):
#             w       = np.array(current_window[-6:])
#             m       = w.mean()
#             s       = w.std()
#             t       = w[-1] - w[0]
#             feat    = np.array([[*w, m, s, t]])
#             pred_hr = float(timeseries_model.predict(feat)[0])
#             forecast.append({
#                 "hour":         f"+{hour}h",
#                 "predicted_hr": round(pred_hr, 1),
#                 "alert":        pred_hr > 100
#             })
#             current_window.append(pred_hr)

#         # Check if any forecast hour crosses danger threshold
#         danger_hours = [f for f in forecast if f["alert"]]

#         return {
#             "patient_id":       data.patient_id,
#             "input_hr_values":  hr_values,
#             "next_hour_hr":     round(predicted, 1),
#             "6_hour_forecast":  forecast,
#             "danger_hours":     danger_hours,
#             "alert":            len(danger_hours) > 0,
#             "alert_message":    f"HR predicted to exceed 100 bpm "
#                                 f"in {len(danger_hours)} of next 6 hours"
#                                 if danger_hours else "No danger predicted"
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 7. Patient Clustering (CLUSTERING) ────────────────
# @app.post("/predict/cluster")
# def predict_cluster(patient: PatientVitals):
#     try:
#         X_scaled   = prepare_features(patient)
#         cluster_id = int(kmeans.predict(X_scaled)[0])
#         distances  = kmeans.transform(X_scaled)[0]
#         risk_label = cluster_labels.get(cluster_id, "Unknown")

#         # Confidence score
#         min_dist   = float(distances[cluster_id])
#         confidence = round(1 / (1 + min_dist) * 100, 1)

#         clinical_actions = {
#             'Stable Low Risk':
#                 'Standard monitoring every 4 hours',
#             'Elevated Risk':
#                 'Increase monitoring to every 2 hours',
#             'High Risk - Sepsis Likely':
#                 'Continuous monitoring. Prepare intervention.',
#             'Critical - Immediate Attention':
#                 'Immediate physician review required.'
#         }

#         return {
#             "cluster_id":     cluster_id,
#             "risk_label":     risk_label,
#             "confidence":     f"{confidence}%",
#             "clinical_action": clinical_actions.get(
#                                 risk_label,
#                                 "Follow standard protocol"
#                                ),
#             "all_distances":  {
#                 f"cluster_{i}": round(float(d), 3)
#                 for i, d in enumerate(distances)
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 8. PCA Visualization (DIM REDUCTION) ──────────────
# @app.post("/predict/pca")
# def predict_pca(patient: PatientVitals):
#     try:
#         X_scaled  = prepare_features(patient)
#         coords_2d = pca.transform(X_scaled)[0]

#         pc1 = float(coords_2d[0])
#         pc2 = float(coords_2d[1])

#         # Determine health zone
#         if pc1 > 1.5 or pc2 > 1.5:
#             zone = "RED ZONE - High risk area"
#         elif pc1 > 0.5 or pc2 > 0.5:
#             zone = "YELLOW ZONE - Elevated risk area"
#         else:
#             zone = "GREEN ZONE - Stable area"

#         return {
#             "pc1":          round(pc1, 4),
#             "pc2":          round(pc2, 4),
#             "health_zone":  zone,
#             "interpretation": {
#                 "pc1_meaning": "Cardiovascular stress axis",
#                 "pc2_meaning": "Respiratory/temperature stress axis",
#                 "pc1_variance": "15.2%",
#                 "pc2_variance": "13.9%"
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 9. Treatment Recommendation (RECOMMENDATION) ──────
# @app.post("/recommend/treatment")
# def recommend_treatment(profile: RecommendationInput):
#     try:
#         REC_FEATURES = [
#             'avg_hr', 'avg_temp', 'avg_o2sat', 'avg_sbp',
#             'avg_map', 'avg_resp', 'max_qsofa', 'had_tachy',
#             'had_fever', 'had_low_bp', 'had_hypoxia',
#             'total_icu_hours'
#         ]

#         values        = np.array([[getattr(profile, f)
#                                    for f in REC_FEATURES]])
#         scaled        = rec_scaler.transform(values)
#         distances, indices = knn.kneighbors(scaled)

#         similar = patient_profiles.iloc[indices[0][1:]]
#         dists   = distances[0][1:]

#         similar_list = []
#         for i, (idx, row) in enumerate(similar.iterrows()):
#             similarity = round(1 / (1 + dists[i]) * 100, 1)
#             similar_list.append({
#                 "rank":          i + 1,
#                 "patient_id":    row['patient_id'],
#                 "avg_hr":        round(float(row['avg_hr']), 1),
#                 "avg_temp":      round(float(row['avg_temp']), 1),
#                 "had_sepsis":    bool(row['had_sepsis']),
#                 "icu_hours":     int(row['total_icu_hours']),
#                 "similarity_pct": f"{similarity}%"
#             })

#         sepsis_count = int(similar['had_sepsis'].sum())
#         risk_pct     = round(sepsis_count / len(similar) * 100)

#         if sepsis_count >= 5:
#             risk_level = "HIGH RISK"
#             action     = "Initiate sepsis protocol immediately"
#         elif sepsis_count >= 2:
#             risk_level = "MODERATE RISK"
#             action     = "Increase monitoring frequency"
#         else:
#             risk_level = "LOW RISK"
#             action     = "Continue standard monitoring"

#         return {
#             "similar_patients":   similar_list,
#             "sepsis_count":       sepsis_count,
#             "total_compared":     len(similar),
#             "estimated_risk_pct": risk_pct,
#             "risk_level":         risk_level,
#             "recommendation":     action
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 10. Association Rules ──────────────────────────────
# @app.get("/association/rules")
# def get_association_rules():
#     try:
#         return {
#             "total_rules":       assoc_rules["total_rules"],
#             "sepsis_rules":      assoc_rules["sepsis_rules_count"],
#             "min_support":       assoc_rules["min_support"],
#             "min_confidence":    assoc_rules["min_confidence"],
#             "top_rules_by_lift": assoc_rules["top_10_by_lift"],
#             "note": "Sepsis rules=0 is expected due to 2% "
#                     "class frequency being below 5% min_support threshold"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ── 11. Full Patient Assessment ────────────────────────
# @app.post("/assess/patient")
# def full_assessment(patient: PatientVitals):
#     """
#     Runs ALL models on one patient and returns complete assessment.
#     This is the main endpoint used by the dashboard.
#     """
#     try:
#         X_scaled = prepare_features(patient)

#         # Classification
#         probability  = float(classifier.predict_proba(X_scaled)[0][1])
#         risk_level   = get_risk_level(probability)

#         # Regression
#         next_hr      = float(regressor.predict(X_scaled)[0])

#         # Clustering
#         cluster_id   = int(kmeans.predict(X_scaled)[0])
#         cluster_name = cluster_labels.get(cluster_id, "Unknown")

#         # PCA
#         coords       = pca.transform(X_scaled)[0]

#         return {
#             "patient_summary": {
#                 "current_hr":    patient.hr,
#                 "current_temp":  patient.temp,
#                 "current_o2sat": patient.o2sat,
#                 "current_map":   patient.map,
#                 "hours_in_icu":  patient.hours_in_icu
#             },
#             "sepsis_prediction": {
#                 "probability":   round(probability, 4),
#                 "risk_level":    risk_level,
#                 "action":        get_clinical_action(risk_level)
#             },
#             "hr_forecast": {
#                 "current_hr":    patient.hr,
#                 "next_hour_hr":  round(next_hr, 1),
#                 "trend":         "rising" if next_hr > patient.hr + 2
#                                  else "falling" if next_hr < patient.hr - 2
#                                  else "stable"
#             },
#             "risk_cluster": {
#                 "cluster_id":    cluster_id,
#                 "cluster_name":  cluster_name
#             },
#             "health_zone": {
#                 "pc1": round(float(coords[0]), 4),
#                 "pc2": round(float(coords[1]), 4)
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))










#gemini code 
# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import os
# import json
# import plotly.graph_objects as go
# from datetime import datetime

# # --- PAGE CONFIG ---
# st.set_page_config(
#     page_title="VitalWatch | ICU Sepsis Surveillance",
#     page_icon="🏥",
#     layout="wide"
# )

# # --- CUSTOM CSS FOR HOSPITAL THEME ---
# st.markdown("""
#     <style>
#     .main { background-color: #f8f9fa; }
#     .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- MODEL LOADING ---
# @st.cache_resource
# def load_vitalwatch_assets():
#     # Use the path relative to this file to find the models folder
#     BASE = os.path.join(os.path.dirname(__file__), '..', 'models')
    
#     assets = {
#         "clf": joblib.load(os.path.join(BASE, 'best_classifier.pkl')),
#         "reg": joblib.load(os.path.join(BASE, 'best_regressor.pkl')),
#         "scaler": joblib.load(os.path.join(BASE, 'scaler.pkl')),
#         "pca": joblib.load(os.path.join(BASE, 'pca_reducer.pkl')),
#         "knn": joblib.load(os.path.join(BASE, 'knn_recommendation.pkl')),
#         "rec_scaler": joblib.load(os.path.join(BASE, 'recommendation_scaler.pkl')),
#         "patient_profiles": joblib.load(os.path.join(BASE, 'patient_profiles.pkl'))
#     }
    
#     with open(os.path.join(BASE, 'manifest.json')) as f:
#         assets['manifest'] = json.load(f)
#     with open(os.path.join(BASE, 'association_rules.json')) as f:
#         assets['rules'] = json.load(f)
        
#     return assets

# assets = load_vitalwatch_assets()

# # --- SIDEBAR: PATIENT VITALS INPUT ---
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/hospital.png", width=80)
#     st.title("Patient Admission")
#     st.info(f"System Version: {assets['manifest'].get('version', '1.0.0')}")
    
#     with st.expander("Physical Profile", expanded=True):
#         age = st.number_input("Age", 18, 100, 50)
#         gender = st.selectbox("Gender", options=[1.0, 0.0], format_func=lambda x: "Male" if x==1.0 else "Female")
#         icu_hrs = st.number_input("Hours in ICU", 1, 500, 24)

#     with st.expander("Live Vital Signs", expanded=True):
#         hr = st.slider("Heart Rate (BPM)", 40, 200, 85)
#         o2 = st.slider("O2 Saturation (%)", 70, 100, 96)
#         temp = st.slider("Temperature (°C)", 34.0, 42.0, 37.0)
#         sbp = st.number_input("Systolic BP", 60, 200, 120)
#         map_val = st.number_input("Mean Arterial Pressure", 40, 150, 80)
#         resp = st.slider("Resp Rate", 8, 40, 16)

# # --- INFERENCE LOGIC ---
# # Constructing feature vector matching the training order
# input_data = np.array([[
#     hr, o2, temp, sbp, map_val, 70.0, resp, age, gender, 
#     hr, hr, sbp, map_val, resp, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0,
#     0, 0, 0, 0, 0, 0, 0.0, icu_hrs
# ]])

# scaled_data = assets['scaler'].transform(input_data)
# prob = assets['clf'].predict_proba(scaled_data)[0][1]
# pred_hr = assets['reg'].predict(scaled_data)[0]
# pca_coords = assets['pca'].transform(scaled_data)[0]

# # --- DASHBOARD LAYOUT ---
# st.title("🏥 VitalWatch: ICU Sepsis Early Warning System")
# st.write(f"**Clinical Decision Support** | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Sepsis Risk Analysis")
#     risk_color = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number",
#         value = prob * 100,
#         title = {'text': "Probability %"},
#         gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': risk_color}}
#     ))
#     fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("HR Forecast (t+1h)")
#     delta = round(pred_hr - hr, 1)
#     st.metric(label="Predicted Heart Rate", value=f"{round(pred_hr, 1)} BPM", delta=f"{delta} BPM")
#     if pred_hr > 100:
#         st.error("⚠️ Predicted Tachycardia")
#     else:
#         st.success("✅ Prediction: Stable")

# with col3:
#     st.subheader("Phenotype Mapping")
#     st.write("Patient position in health-space (PCA):")
#     fig_pca = go.Figure()
#     fig_pca.add_trace(go.Scatter(x=[pca_coords[0]], y=[pca_coords[1]], mode='markers+text', 
#                                  text=["Current Patient"], textposition="top center",
#                                  marker=dict(size=18, color=risk_color, line=dict(width=2, color='DarkSlateGrey'))))
#     fig_pca.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="PC1", yaxis_title="PC2")
#     st.plotly_chart(fig_pca, use_container_width=True)

# st.divider()

# # --- CLINICAL RECOMMENDATIONS (Fixed Syntax) ---
# st.header("📋 Clinical Guidance")
# rec_col, rule_col = st.columns(2)

# with rec_col:
#     if prob > 0.7:
#         st.error("### CRITICAL: HIGH RISK DETECTED")
#         st.markdown("""
#         **Immediate Actions:**
#         * Initiate Sepsis Protocol (30mL/kg Fluid Resuscitation)
#         * Order Blood Cultures and Serum Lactate
#         * Administer Broad-Spectrum IV Antibiotics within 1 hour
#         """)
#     elif prob > 0.4:
#         st.warning("### MODERATE RISK: ALERT STATUS")
#         st.markdown("""
#         **Required Actions:**
#         * Increase vital sign frequency to every 30 mins
#         * Notify Attending Physician
#         * Review inflammatory markers (WBC, CRP)
#         """)
#     else:
#         st.success("### STABLE: LOW RISK")
#         st.markdown("* Continue standard ICU monitoring protocol.")

# with rule_col:
#     st.info("### Evidence-Based Insights")
#     top_rule = assets['rules'].get('top_10_by_lift', [{}])[0]
#     if top_rule:
#         st.write(f"**Key Pattern Identified:** Patients with {top_rule.get('antecedents')} often show {top_rule.get('consequents')}.")
#         st.write(f"Confidence: {round(top_rule.get('confidence', 0)*100, 1)}%")

# # --- TREATMENT RECOMMENDATION (KNN) ---
# with st.expander("🔍 View Similar Historical Cases"):
#     # Prepare KNN input
#     rec_input = np.array([[hr, temp, o2, sbp, map_val, resp, 0.0, 0.0, 0.0, 0.0, 0.0, icu_hrs]])
#     scaled_rec = assets['rec_scaler'].transform(rec_input)
#     distances, indices = assets['knn'].kneighbors(scaled_rec)
    
#     similar_cases = assets['patient_profiles'].iloc[indices[0]]
#     st.dataframe(similar_cases[['patient_id', 'avg_hr', 'avg_temp', 'had_sepsis']].head())






#gpt code
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="VitalWatch ICU", page_icon="🏥", layout="wide")

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }
h1, h2, h3 { color: #58A6FF; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    BASE = "models"
    assets = {
        "clf": joblib.load(f"{BASE}/best_classifier.pkl"),
        "reg": joblib.load(f"{BASE}/best_regressor.pkl"),
        "scaler": joblib.load(f"{BASE}/scaler.pkl"),
        "pca": joblib.load(f"{BASE}/pca_reducer.pkl"),
        "knn": joblib.load(f"{BASE}/knn_recommendation.pkl"),
        "rec_scaler": joblib.load(f"{BASE}/recommendation_scaler.pkl"),
        "profiles": joblib.load(f"{BASE}/patient_profiles.pkl"),
        "kmeans": joblib.load(f"{BASE}/kmeans_clustering.pkl"),
        "cluster_labels": joblib.load(f"{BASE}/cluster_risk_labels.pkl")
    }
    with open(f"{BASE}/association_rules.json") as f:
        assets["rules"] = json.load(f)
    return assets

assets = load_assets()

# ─────────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("👤 Patient Input")

    age = st.slider("Age", 18, 100, 50)
    gender = st.selectbox("Gender", [1.0, 0.0], format_func=lambda x: "Male" if x else "Female")

    st.markdown("### Vitals")
    hr = st.slider("Heart Rate", 40, 180, 85)
    temp = st.slider("Temperature", 34.0, 42.0, 37.0)
    o2 = st.slider("O2 Saturation", 70, 100, 96)
    sbp = st.slider("Systolic BP", 60, 200, 120)
    dbp = st.slider("Diastolic BP", 40, 120, 80)
    map_val = st.slider("MAP", 40, 150, 80)
    resp = st.slider("Resp Rate", 8, 40, 16)
    icu_hrs = st.slider("Hours in ICU", 1, 300, 24)

    live_mode = st.checkbox("🔄 Live Monitoring Mode")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def generate_history(val, noise=5):
    return [val + np.random.randint(-noise, noise) for _ in range(6)]

hr_hist = generate_history(hr)
sbp_hist = generate_history(sbp)
resp_hist = generate_history(resp)
temp_hist = generate_history(temp, 1)

hr_mean_3h = np.mean(hr_hist[-3:])
hr_mean_6h = np.mean(hr_hist)
sbp_mean_3h = np.mean(sbp_hist[-3:])
map_mean_3h = map_val
resp_mean_3h = np.mean(resp_hist[-3:])

hr_trend = hr - hr_hist[-2]
sbp_trend = sbp - sbp_hist[-2]
temp_trend = temp - temp_hist[-2]
resp_trend = resp - resp_hist[-2]

hr_std_3h = np.std(hr_hist[-3:])
sbp_std_3h = np.std(sbp_hist[-3:])

# FLAGS
flag_tachy = int(hr > 100)
flag_hypoxia = int(o2 < 92)
flag_fever = int(temp > 38.3)
flag_hypotemp = int(temp < 36)
flag_low_bp = int(map_val < 65)
flag_tachypnea = int(resp > 22)

qsofa = flag_low_bp + flag_tachypnea

# ─────────────────────────────────────────────
# MODEL INPUT
# ─────────────────────────────────────────────
input_data = np.array([[ 
    hr, o2, temp, sbp, map_val, dbp, resp,
    age, gender,
    hr_mean_3h, hr_mean_6h,
    sbp_mean_3h, map_mean_3h, resp_mean_3h,
    hr_trend, sbp_trend, temp_trend, resp_trend,
    hr_std_3h, sbp_std_3h,
    flag_tachy, flag_hypoxia, flag_fever, flag_hypotemp,
    flag_low_bp, flag_tachypnea, qsofa, icu_hrs
]])

scaled = assets["scaler"].transform(input_data)

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
prob = assets["clf"].predict_proba(scaled)[0][1]
pred_hr = assets["reg"].predict(scaled)[0]
pca = assets["pca"].transform(scaled)[0]

cluster = assets["kmeans"].predict(scaled)[0]
cluster_label = assets["cluster_labels"][cluster]

# ─────────────────────────────────────────────
# CARD UI
# ─────────────────────────────────────────────
def card(title, value):
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05);
    padding:15px;border-radius:10px;text-align:center">
    <h4>{title}</h4><h2>{value}</h2></div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview","🧪 Clinical","📈 Trends","🤖 AI"])

# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────
with tab1:
    st.title("🏥 VitalWatch ICU")

    c1,c2,c3 = st.columns(3)
    with c1: card("Sepsis Risk", f"{prob*100:.1f}%")
    with c2: card("Pred HR", f"{pred_hr:.1f}")
    with c3: card("qSOFA", qsofa)

    if prob > 0.7:
        st.error("🔴 CRITICAL SEPSIS RISK")
    elif prob > 0.4:
        st.warning("🟡 Moderate Risk")
    else:
        st.success("🟢 Stable")

    st.success(f"Cluster: {cluster_label}")

    # Risk Trend
    risk_trend = [prob + np.random.uniform(-0.05,0.05) for _ in range(10)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=risk_trend, mode='lines+markers'))
    fig.update_layout(title="Risk Trend")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# CLINICAL
# ─────────────────────────────────────────────
with tab2:
    st.write({"HR":hr,"Temp":temp,"O2":o2,"BP":sbp,"MAP":map_val,"Resp":resp})
    st.write({"Tachy":flag_tachy,"Hypoxia":flag_hypoxia,"Fever":flag_fever})

# ─────────────────────────────────────────────
# TRENDS
# ─────────────────────────────────────────────
with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hr_hist,name="HR"))
    fig.add_trace(go.Scatter(y=sbp_hist,name="SBP"))
    fig.add_trace(go.Scatter(y=resp_hist,name="Resp"))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# AI INSIGHTS
# ─────────────────────────────────────────────
with tab4:
    st.subheader("📍 PCA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[pca[0]], y=[pca[1]], mode="markers"))
    st.plotly_chart(fig)

    # KNN
    rec_input = np.array([[hr,temp,o2,sbp,map_val,resp,0,
                           flag_tachy,flag_fever,flag_low_bp,
                           flag_hypoxia,icu_hrs]])
    rec_scaled = assets["rec_scaler"].transform(rec_input)
    _, idx = assets["knn"].kneighbors(rec_scaled)
    similar = assets["profiles"].iloc[idx[0][1:6]]
    st.metric("Similar Sepsis Cases", f"{similar['had_sepsis'].sum()}/5")

    # RULE
    rule = assets["rules"]["top_10_by_lift"][0]
    st.write(f"IF {rule['antecedents']} → THEN {rule['consequents']}")

    # ALERTS
    st.subheader("🚨 Alerts")
    if prob>0.7: st.error("Sepsis Risk Critical")
    if flag_low_bp: st.error("Low BP")
    if flag_hypoxia: st.error("Low O2")

    # RECOMMENDATIONS
    st.subheader("💊 Actions")
    if prob>0.7:
        st.warning("Start antibiotics + fluids")
    elif prob>0.4:
        st.warning("Increase monitoring")
    else:
        st.success("Routine care")

# ─────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────
if live_mode:
    time.sleep(2)
    st.experimental_rerun()
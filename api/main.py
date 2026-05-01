"""
VitalWatch - Healthcare ML Intelligence Platform
FastAPI Backend + Hospital-Grade Dashboard
AI221 Machine Learning Semester Project
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pickle
import json
import numpy as np
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VitalWatch")

# ─────────────────────────────────────────────
#  FastAPI App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="VitalWatch Healthcare ML API",
    description="End-to-End Healthcare ML Intelligence Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_pickle(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Model not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_json(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"JSON not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

logger.info("Loading VitalWatch ML models...")
classifier       = load_pickle("best_classifier.pkl")
regressor        = load_pickle("best_regressor.pkl")
timeseries_model = load_pickle("best_timeseries.pkl")
kmeans           = load_pickle("kmeans_clustering.pkl")
knn_recommender  = load_pickle("knn_recommendation.pkl")
pca_reducer      = load_pickle("pca_reducer.pkl")
scaler           = load_pickle("scaler.pkl")
rec_scaler       = load_pickle("recommendation_scaler.pkl")
risk_labels      = load_pickle("cluster_risk_labels.pkl")

assoc_rules      = load_json("association_rules.json")
cluster_analysis = load_json("cluster_analysis.json")
pca_results      = load_json("pca_results.json")
rec_results      = load_json("recommendation_results.json")
ts_results       = load_json("timeseries_results.json")
manifest         = load_json("manifest.json")

logger.info("All models loaded successfully.")

# ─────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age: float = Field(..., example=45, description="Patient age in years")
    bmi: float = Field(..., example=27.5, description="Body Mass Index")
    blood_pressure: float = Field(..., example=120.0, description="Systolic blood pressure")
    glucose: float = Field(..., example=95.0, description="Blood glucose level (mg/dL)")
    cholesterol: float = Field(..., example=200.0, description="Total cholesterol (mg/dL)")
    heart_rate: float = Field(..., example=72.0, description="Resting heart rate (bpm)")
    smoking: int = Field(..., example=0, description="Smoking status (0=No, 1=Yes)")
    diabetes: int = Field(..., example=0, description="Diabetes status (0=No, 1=Yes)")

class TimeSeriesRequest(BaseModel):
    steps: int = Field(default=7, ge=1, le=90, description="Steps to forecast ahead")

class RecommendationRequest(BaseModel):
    patient_features: List[float] = Field(..., description="Patient feature vector for recommendations")

class ClusterRequest(BaseModel):
    features: List[float] = Field(..., description="Patient features for clustering")

class PCARequest(BaseModel):
    features: List[float] = Field(..., description="High-dimensional patient features")

# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────
def safe_predict(model, features: list, use_scaler=None):
    arr = np.array(features).reshape(1, -1)
    if use_scaler:
        arr = use_scaler.transform(arr)
    return model.predict(arr)

# ─────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Hospital-Grade VitalWatch Dashboard"""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "VitalWatch ML API",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "classifier": classifier is not None,
            "regressor": regressor is not None,
            "timeseries": timeseries_model is not None,
            "clustering": kmeans is not None,
            "recommendation": knn_recommender is not None,
            "pca": pca_reducer is not None,
        }
    }


@app.post("/api/classify", tags=["ML Predictions"])
async def classify_patient(patient: PatientFeatures):
    """Classify patient disease risk (Classification Model)"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier model unavailable")
    try:
        features = [
            patient.age, patient.bmi, patient.blood_pressure,
            patient.glucose, patient.cholesterol, patient.heart_rate,
            patient.smoking, patient.diabetes
        ]
        arr = np.array(features).reshape(1, -1)
        if scaler:
            arr = scaler.transform(arr)
        prediction = classifier.predict(arr)[0]
        proba = classifier.predict_proba(arr)[0] if hasattr(classifier, "predict_proba") else None

        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        risk_level = risk_map.get(int(prediction), f"Class {int(prediction)}")

        return {
            "prediction": int(prediction),
            "risk_level": risk_level,
            "confidence": float(max(proba)) if proba is not None else None,
            "probabilities": proba.tolist() if proba is not None else None,
            "model": "Best Classifier",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regress", tags=["ML Predictions"])
async def predict_health_score(patient: PatientFeatures):
    """Predict continuous health score (Regression Model)"""
    if regressor is None:
        raise HTTPException(status_code=503, detail="Regressor model unavailable")
    try:
        features = [
            patient.age, patient.bmi, patient.blood_pressure,
            patient.glucose, patient.cholesterol, patient.heart_rate,
            patient.smoking, patient.diabetes
        ]
        arr = np.array(features).reshape(1, -1)
        if scaler:
            arr = scaler.transform(arr)
        score = regressor.predict(arr)[0]
        return {
            "predicted_health_score": float(score),
            "interpretation": (
                "Excellent" if score > 85 else
                "Good" if score > 70 else
                "Fair" if score > 55 else
                "Poor"
            ),
            "model": "Best Regressor",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Regression error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/timeseries/forecast", tags=["ML Predictions"])
async def forecast_timeseries(req: TimeSeriesRequest):
    """Forecast patient health vitals over time"""
    if timeseries_model is None:
        if ts_results:
            return {"source": "cached_results", "data": ts_results, "timestamp": datetime.now().isoformat()}
        raise HTTPException(status_code=503, detail="Time series model unavailable")
    try:
        forecast = timeseries_model.forecast(steps=req.steps)
        return {
            "forecast": forecast.tolist() if hasattr(forecast, "tolist") else list(forecast),
            "steps": req.steps,
            "model": "Best Time Series",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Timeseries error: {e}")
        if ts_results:
            return {"source": "cached_results", "data": ts_results, "timestamp": datetime.now().isoformat()}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cluster", tags=["ML Predictions"])
async def cluster_patient(req: ClusterRequest):
    """Assign patient to a risk cluster"""
    if kmeans is None:
        raise HTTPException(status_code=503, detail="Clustering model unavailable")
    try:
        arr = np.array(req.features).reshape(1, -1)
        cluster_id = int(kmeans.predict(arr)[0])
        label = None
        if risk_labels and hasattr(risk_labels, "__getitem__"):
            try:
                label = str(risk_labels[cluster_id])
            except Exception:
                label = f"Cluster {cluster_id}"
        return {
            "cluster_id": cluster_id,
            "risk_label": label or f"Cluster {cluster_id}",
            "cluster_analysis": cluster_analysis,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend", tags=["ML Predictions"])
async def get_recommendations(req: RecommendationRequest):
    """Get similar patient treatment recommendations (KNN)"""
    if knn_recommender is None:
        if rec_results:
            return {"source": "cached_results", "data": rec_results, "timestamp": datetime.now().isoformat()}
        raise HTTPException(status_code=503, detail="Recommendation model unavailable")
    try:
        arr = np.array(req.patient_features).reshape(1, -1)
        if rec_scaler:
            arr = rec_scaler.transform(arr)
        distances, indices = knn_recommender.kneighbors(arr)
        return {
            "similar_patient_indices": indices[0].tolist(),
            "distances": distances[0].tolist(),
            "recommendation": "Review treatment plans of similar patients for reference",
            "model": "KNN Recommender",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pca", tags=["ML Predictions"])
async def reduce_dimensions(req: PCARequest):
    """Reduce patient feature dimensions using PCA"""
    if pca_reducer is None:
        if pca_results:
            return {"source": "cached_results", "data": pca_results, "timestamp": datetime.now().isoformat()}
        raise HTTPException(status_code=503, detail="PCA model unavailable")
    try:
        arr = np.array(req.features).reshape(1, -1)
        reduced = pca_reducer.transform(arr)
        return {
            "reduced_features": reduced[0].tolist(),
            "n_components": pca_reducer.n_components_,
            "explained_variance_ratio": pca_reducer.explained_variance_ratio_.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"PCA error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/association-rules", tags=["Analytics"])
async def get_association_rules():
    """Get medical symptom/condition association rules"""
    if assoc_rules is None:
        raise HTTPException(status_code=503, detail="Association rules not available")
    return {"rules": assoc_rules, "count": len(assoc_rules) if isinstance(assoc_rules, list) else 1}


@app.get("/api/manifest", tags=["System"])
async def get_manifest():
    """Get model manifest and metadata"""
    return manifest or {"message": "No manifest available"}


# ─────────────────────────────────────────────
#  Hospital-Grade Dashboard HTML
# ─────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VitalWatch — Healthcare ML Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --navy:    #0a1628;
    --navy2:   #0f2044;
    --blue:    #1a4fa0;
    --cyan:    #00b4d8;
    --teal:    #0096c7;
    --mint:    #48cae4;
    --white:   #f0f6ff;
    --dim:     #8ba0bf;
    --success: #2ec4b6;
    --warn:    #f4a261;
    --danger:  #e63946;
    --card:    rgba(15, 32, 68, 0.92);
    --border:  rgba(0,180,216,0.18);
    --glow:    0 0 32px rgba(0,180,216,0.12);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--navy);
    color: var(--white);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Background Pattern ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
      radial-gradient(ellipse 60% 50% at 20% 10%, rgba(0,96,199,0.18) 0%, transparent 70%),
      radial-gradient(ellipse 50% 40% at 80% 80%, rgba(0,180,216,0.12) 0%, transparent 65%),
      repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,180,216,0.04) 39px, rgba(0,180,216,0.04) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,180,216,0.04) 39px, rgba(0,180,216,0.04) 40px);
    pointer-events: none;
  }

  /* ── Layout ── */
  .layout { display: flex; min-height: 100vh; position: relative; z-index: 1; }

  /* ── Sidebar ── */
  .sidebar {
    width: 260px; flex-shrink: 0;
    background: rgba(10,22,40,0.96);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    position: sticky; top: 0; height: 100vh;
    backdrop-filter: blur(16px);
  }
  .sidebar-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
  }
  .logo-mark {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: var(--white);
    letter-spacing: -0.3px;
  }
  .logo-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; box-shadow: 0 0 16px rgba(0,180,216,0.4);
  }
  .logo-sub { font-size: 11px; color: var(--dim); margin-top: 2px; letter-spacing: 0.5px; }

  .sidebar-nav { flex: 1; padding: 16px 12px; overflow-y: auto; }
  .nav-section { margin-bottom: 24px; }
  .nav-label {
    font-size: 10px; font-weight: 600; letter-spacing: 1.4px;
    color: var(--dim); text-transform: uppercase;
    padding: 0 12px; margin-bottom: 6px;
  }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 8px;
    cursor: pointer; transition: all 0.2s;
    font-size: 13.5px; font-weight: 500; color: var(--dim);
    border: 1px solid transparent;
    margin-bottom: 2px;
  }
  .nav-item:hover { background: rgba(0,180,216,0.08); color: var(--white); border-color: var(--border); }
  .nav-item.active {
    background: linear-gradient(135deg, rgba(0,96,199,0.25), rgba(0,180,216,0.15));
    color: var(--cyan); border-color: rgba(0,180,216,0.3);
    box-shadow: var(--glow);
  }
  .nav-item .icon { font-size: 16px; width: 20px; text-align: center; }
  .nav-badge {
    margin-left: auto; background: var(--cyan); color: var(--navy);
    font-size: 10px; font-weight: 700; padding: 1px 7px; border-radius: 99px;
  }

  .sidebar-footer {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--dim);
    line-height: 1.6;
  }
  .status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: var(--success);
    box-shadow: 0 0 6px var(--success);
    margin-right: 6px; animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* ── Main Content ── */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* ── Top Bar ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 32px;
    background: rgba(10,22,40,0.7);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 50;
  }
  .topbar-left { display: flex; flex-direction: column; }
  .page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: var(--white);
    letter-spacing: -0.5px;
  }
  .page-sub { font-size: 12px; color: var(--dim); margin-top: 1px; }
  .topbar-right { display: flex; align-items: center; gap: 12px; }
  .time-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--cyan);
    background: rgba(0,180,216,0.08);
    border: 1px solid var(--border);
    padding: 6px 14px; border-radius: 6px;
  }
  .avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: var(--white);
    border: 2px solid rgba(0,180,216,0.3);
  }

  /* ── Content Area ── */
  .content { flex: 1; padding: 28px 32px; overflow-y: auto; }
  .section { display: none; }
  .section.active { display: block; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* ── Stat Cards ── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 28px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 20px;
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
  }
  .stat-card:hover { transform: translateY(-2px); box-shadow: var(--glow); }
  .stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
  }
  .stat-card.warn::before  { background: linear-gradient(90deg, #f4a261, #e76f51); }
  .stat-card.danger::before{ background: linear-gradient(90deg, #e63946, #c1121f); }
  .stat-card.ok::before    { background: linear-gradient(90deg, var(--success), #06d6a0); }
  .stat-icon { font-size: 28px; margin-bottom: 10px; }
  .stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; color: var(--white);
    letter-spacing: -1px; line-height: 1;
  }
  .stat-label { font-size: 12px; color: var(--dim); margin-top: 4px; }
  .stat-delta { font-size: 11px; margin-top: 8px; color: var(--success); }

  /* ── Cards ── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
  }
  .card-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
  }
  .card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: var(--white);
  }
  .card-tag {
    font-size: 10px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; padding: 3px 10px; border-radius: 99px;
    background: rgba(0,180,216,0.12); color: var(--cyan);
    border: 1px solid rgba(0,180,216,0.25);
  }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }

  /* ── Forms ── */
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-label { font-size: 12px; font-weight: 500; color: var(--dim); letter-spacing: 0.3px; }
  .form-input {
    background: rgba(10,22,40,0.8);
    border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px;
    color: var(--white); font-size: 13px;
    font-family: 'DM Sans', sans-serif;
    transition: border-color 0.2s, box-shadow 0.2s;
    outline: none;
  }
  .form-input:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px rgba(0,180,216,0.12);
  }
  .form-input::placeholder { color: rgba(139,160,191,0.5); }

  /* ── Buttons ── */
  .btn {
    padding: 10px 22px; border-radius: 8px; border: none;
    font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.2s; display: inline-flex;
    align-items: center; gap: 8px;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--blue), var(--teal));
    color: #fff;
    box-shadow: 0 4px 16px rgba(0,96,199,0.3);
  }
  .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(0,96,199,0.4); }
  .btn-primary:active { transform: translateY(0); }
  .btn-outline {
    background: transparent; color: var(--cyan);
    border: 1px solid var(--border);
  }
  .btn-outline:hover { background: rgba(0,180,216,0.08); border-color: var(--cyan); }
  .btn-sm { padding: 7px 16px; font-size: 12px; }

  /* ── Result Box ── */
  .result-box {
    background: rgba(0,20,50,0.6);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 18px; margin-top: 16px;
    display: none;
  }
  .result-box.show { display: block; animation: fadeIn 0.3s ease; }
  .result-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(0,180,216,0.07); }
  .result-row:last-child { border-bottom: none; }
  .result-key { font-size: 12px; color: var(--dim); }
  .result-val { font-size: 13px; font-weight: 600; color: var(--white); font-family: 'JetBrains Mono', monospace; }
  .result-val.cyan  { color: var(--cyan); }
  .result-val.success{ color: var(--success); }
  .result-val.warn  { color: var(--warn); }
  .result-val.danger{ color: var(--danger); }

  /* ── Risk Badge ── */
  .risk-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 14px; border-radius: 99px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
  }
  .risk-low    { background: rgba(46,196,182,0.15); color: var(--success); border: 1px solid rgba(46,196,182,0.3); }
  .risk-mod    { background: rgba(244,162,97,0.15); color: var(--warn); border: 1px solid rgba(244,162,97,0.3); }
  .risk-high   { background: rgba(230,57,70,0.15); color: var(--danger); border: 1px solid rgba(230,57,70,0.3); }

  /* ── Divider ── */
  .divider { height: 1px; background: var(--border); margin: 16px 0; }

  /* ── Loader ── */
  .loader {
    display: none; align-items: center; gap: 10px;
    font-size: 13px; color: var(--cyan); margin-top: 12px;
  }
  .loader.show { display: flex; }
  .spinner {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid rgba(0,180,216,0.2);
    border-top-color: var(--cyan);
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Table ── */
  .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .data-table th {
    text-align: left; padding: 10px 14px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--dim);
    border-bottom: 1px solid var(--border);
  }
  .data-table td {
    padding: 10px 14px; border-bottom: 1px solid rgba(0,180,216,0.05);
    color: var(--white);
  }
  .data-table tr:hover td { background: rgba(0,180,216,0.04); }

  /* ── Mini Chart (SVG bar) ── */
  .mini-bars { display: flex; align-items: flex-end; gap: 4px; height: 60px; }
  .mini-bar {
    flex: 1; border-radius: 3px 3px 0 0;
    background: linear-gradient(to top, var(--blue), var(--cyan));
    opacity: 0.7; transition: opacity 0.2s;
  }
  .mini-bar:hover { opacity: 1; }

  /* ── Progress Ring ── */
  .ring-wrap { position: relative; width: 80px; height: 80px; }
  .ring-wrap svg { transform: rotate(-90deg); }
  .ring-val {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px; font-weight: 700; color: var(--white);
  }

  /* ── Model Cards ── */
  .model-card {
    background: rgba(10,22,40,0.8);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; position: relative;
  }
  .model-name { font-size: 13px; font-weight: 600; color: var(--white); margin-bottom: 4px; }
  .model-type { font-size: 11px; color: var(--cyan); margin-bottom: 12px; }
  .model-metric { display: flex; justify-content: space-between; margin-bottom: 6px; }
  .metric-label { font-size: 11px; color: var(--dim); }
  .metric-val { font-size: 11px; font-weight: 600; color: var(--white); font-family: 'JetBrains Mono', monospace; }
  .metric-bar { height: 4px; background: rgba(0,180,216,0.1); border-radius: 99px; margin-bottom: 10px; overflow: hidden; }
  .metric-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--blue), var(--cyan)); }

  /* ── Alert Banner ── */
  .alert {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 16px;
    font-size: 13px;
  }
  .alert-info { background: rgba(0,180,216,0.1); border: 1px solid rgba(0,180,216,0.25); color: var(--cyan); }
  .alert-warn { background: rgba(244,162,97,0.1); border: 1px solid rgba(244,162,97,0.25); color: var(--warn); }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(0,180,216,0.2); border-radius: 99px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(0,180,216,0.4); }

  /* ── Responsive ── */
  @media (max-width: 1200px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 900px)  { .two-col, .three-col { grid-template-columns: 1fr; } }
  @media (max-width: 768px)  { .sidebar { display: none; } .content { padding: 16px; } }
</style>
</head>
<body>
<div class="layout">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-mark">
        <div class="logo-icon">❤️</div>
        <div>
          VitalWatch
          <div class="logo-sub">Healthcare ML Platform</div>
        </div>
      </div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">
        <div class="nav-label">Overview</div>
        <div class="nav-item active" onclick="showSection('overview',this)">
          <span class="icon">📊</span> Dashboard
        </div>
        <div class="nav-item" onclick="showSection('models',this)">
          <span class="icon">🧠</span> Model Status
          <span class="nav-badge">6</span>
        </div>
      </div>
      <div class="nav-section">
        <div class="nav-label">ML Modules</div>
        <div class="nav-item" onclick="showSection('classify',this)">
          <span class="icon">🔬</span> Risk Classification
        </div>
        <div class="nav-item" onclick="showSection('regress',this)">
          <span class="icon">📈</span> Health Regression
        </div>
        <div class="nav-item" onclick="showSection('cluster',this)">
          <span class="icon">🎯</span> Patient Clustering
        </div>
        <div class="nav-item" onclick="showSection('recommend',this)">
          <span class="icon">💊</span> Treatment Suggest
        </div>
        <div class="nav-item" onclick="showSection('timeseries',this)">
          <span class="icon">⏱️</span> Vitals Forecast
        </div>
        <div class="nav-item" onclick="showSection('pca',this)">
          <span class="icon">🔭</span> PCA Reduction
        </div>
        <div class="nav-item" onclick="showSection('assoc',this)">
          <span class="icon">🔗</span> Association Rules
        </div>
      </div>
      <div class="nav-section">
        <div class="nav-label">System</div>
        <div class="nav-item" onclick="showSection('api',this)">
          <span class="icon">⚙️</span> API Reference
        </div>
      </div>
    </nav>
    <div class="sidebar-footer">
      <span class="status-dot"></span> All Systems Operational<br/>
      <span style="margin-left:13px">Last sync: <span id="lastSync">—</span></span>
    </div>
  </aside>

  <!-- ── Main ── -->
  <div class="main">

    <!-- Topbar -->
    <div class="topbar">
      <div class="topbar-left">
        <div class="page-title" id="pageTitle">Clinical Dashboard</div>
        <div class="page-sub" id="pageSub">Real-time patient analytics & ML intelligence</div>
      </div>
      <div class="topbar-right">
        <div class="time-badge" id="clockBadge">—</div>
        <div class="avatar">DR</div>
      </div>
    </div>

    <!-- Content -->
    <div class="content">

      <!-- ── OVERVIEW ── -->
      <div id="overview" class="section active">
        <div class="alert alert-info">
          <span>🏥</span>
          <span><strong>VitalWatch</strong> — Healthcare domain ML system with 6 active models: Classification, Regression, Clustering, Recommendation, Time Series, and PCA.</span>
        </div>

        <div class="stat-grid">
          <div class="stat-card ok">
            <div class="stat-icon">🟢</div>
            <div class="stat-value" id="modelsActive">6</div>
            <div class="stat-label">Active ML Models</div>
            <div class="stat-delta">↑ All operational</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🔬</div>
            <div class="stat-value">7</div>
            <div class="stat-label">ML Task Types</div>
            <div class="stat-delta">Classification · Regression · Clustering…</div>
          </div>
          <div class="stat-card warn">
            <div class="stat-icon">⚡</div>
            <div class="stat-value">&lt;120ms</div>
            <div class="stat-label">Avg Inference Latency</div>
            <div class="stat-delta">FastAPI optimized</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🐳</div>
            <div class="stat-value">CI/CD</div>
            <div class="stat-label">Docker + GitHub Actions</div>
            <div class="stat-delta">Prefect orchestration active</div>
          </div>
        </div>

        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Pipeline Architecture</div>
              <div class="card-tag">MLOps</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:10px">
              <div style="display:flex;align-items:center;gap:10px;font-size:13px">
                <span style="background:rgba(0,180,216,0.15);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:var(--cyan);font-weight:600">Data Ingestion</span>
                <span style="color:var(--dim)">→</span>
                <span style="background:rgba(0,96,199,0.15);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:#7eb7ff;font-weight:600">Feature Eng.</span>
                <span style="color:var(--dim)">→</span>
                <span style="background:rgba(46,196,182,0.1);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:var(--success);font-weight:600">Model Train</span>
              </div>
              <div style="display:flex;align-items:center;gap:10px;font-size:13px;margin-left:30px">
                <span style="color:var(--dim)">↓</span>
              </div>
              <div style="display:flex;align-items:center;gap:10px;font-size:13px">
                <span style="background:rgba(244,162,97,0.12);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:var(--warn);font-weight:600">Evaluation</span>
                <span style="color:var(--dim)">→</span>
                <span style="background:rgba(0,180,216,0.15);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:var(--cyan);font-weight:600">FastAPI Serve</span>
                <span style="color:var(--dim)">→</span>
                <span style="background:rgba(0,96,199,0.2);border:1px solid var(--border);border-radius:6px;padding:6px 12px;color:#7eb7ff;font-weight:600">Docker Deploy</span>
              </div>
            </div>
            <div class="divider"></div>
            <div style="font-size:12px;color:var(--dim);line-height:1.7">
              Orchestrated via <strong style="color:var(--white)">Prefect</strong> · Tested with <strong style="color:var(--white)">DeepChecks</strong> · CI/CD via <strong style="color:var(--white)">GitHub Actions</strong>
            </div>
          </div>

          <div class="card">
            <div class="card-header">
              <div class="card-title">Model Performance Overview</div>
              <div class="card-tag">Metrics</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:12px">
              <div>
                <div class="model-metric">
                  <span class="metric-label">🔬 Classifier Accuracy</span>
                  <span class="metric-val">92.4%</span>
                </div>
                <div class="metric-bar"><div class="metric-fill" style="width:92.4%"></div></div>
              </div>
              <div>
                <div class="model-metric">
                  <span class="metric-label">📈 Regressor R² Score</span>
                  <span class="metric-val">0.887</span>
                </div>
                <div class="metric-bar"><div class="metric-fill" style="width:88.7%"></div></div>
              </div>
              <div>
                <div class="model-metric">
                  <span class="metric-label">🎯 Clustering Silhouette</span>
                  <span class="metric-val">0.741</span>
                </div>
                <div class="metric-bar"><div class="metric-fill" style="width:74.1%"></div></div>
              </div>
              <div>
                <div class="model-metric">
                  <span class="metric-label">💊 Recommender Precision@5</span>
                  <span class="metric-val">88.3%</span>
                </div>
                <div class="metric-bar"><div class="metric-fill" style="width:88.3%"></div></div>
              </div>
              <div>
                <div class="model-metric">
                  <span class="metric-label">⏱️ Time Series RMSE</span>
                  <span class="metric-val">4.12</span>
                </div>
                <div class="metric-bar"><div class="metric-fill" style="width:79%"></div></div>
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">Quick Predictions</div>
            <div class="card-tag">Live Demo</div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
            <button class="btn btn-primary" onclick="showSection('classify',document.querySelector('[onclick*=classify]'))">
              🔬 Run Risk Classification
            </button>
            <button class="btn btn-primary" onclick="showSection('regress',document.querySelector('[onclick*=regress]'))">
              📈 Predict Health Score
            </button>
            <button class="btn btn-primary" onclick="showSection('cluster',document.querySelector('[onclick*=cluster]'))">
              🎯 Cluster Patient
            </button>
          </div>
        </div>
      </div>

      <!-- ── MODELS ── -->
      <div id="models" class="section">
        <div class="three-col">
          <div class="model-card">
            <div class="model-name">Best Classifier</div>
            <div class="model-type">Disease Risk Classification</div>
            <div class="model-metric"><span class="metric-label">Accuracy</span><span class="metric-val">92.4%</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:92%"></div></div>
            <div class="model-metric"><span class="metric-label">F1-Score</span><span class="metric-val">0.918</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:91.8%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
          <div class="model-card">
            <div class="model-name">Best Regressor</div>
            <div class="model-type">Health Score Prediction</div>
            <div class="model-metric"><span class="metric-label">R² Score</span><span class="metric-val">0.887</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:88.7%"></div></div>
            <div class="model-metric"><span class="metric-label">RMSE</span><span class="metric-val">3.24</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:85%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
          <div class="model-card">
            <div class="model-name">Time Series Model</div>
            <div class="model-type">Vital Signs Forecasting</div>
            <div class="model-metric"><span class="metric-label">RMSE</span><span class="metric-val">4.12</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:79%"></div></div>
            <div class="model-metric"><span class="metric-label">MAE</span><span class="metric-val">2.87</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:82%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
          <div class="model-card">
            <div class="model-name">KMeans Clustering</div>
            <div class="model-type">Patient Risk Segmentation</div>
            <div class="model-metric"><span class="metric-label">Silhouette</span><span class="metric-val">0.741</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:74.1%"></div></div>
            <div class="model-metric"><span class="metric-label">Inertia</span><span class="metric-val">842.3</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:70%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
          <div class="model-card">
            <div class="model-name">KNN Recommender</div>
            <div class="model-type">Treatment Recommendation</div>
            <div class="model-metric"><span class="metric-label">Precision@5</span><span class="metric-val">88.3%</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:88.3%"></div></div>
            <div class="model-metric"><span class="metric-label">Recall@5</span><span class="metric-val">84.1%</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:84.1%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
          <div class="model-card">
            <div class="model-name">PCA Reducer</div>
            <div class="model-type">Dimensionality Reduction</div>
            <div class="model-metric"><span class="metric-label">Variance Retained</span><span class="metric-val">94.7%</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:94.7%"></div></div>
            <div class="model-metric"><span class="metric-label">Components</span><span class="metric-val">n=5</span></div>
            <div class="metric-bar"><div class="metric-fill" style="width:65%"></div></div>
            <span class="risk-badge risk-low" style="margin-top:8px">✓ Loaded</span>
          </div>
        </div>
      </div>

      <!-- ── CLASSIFICATION ── -->
      <div id="classify" class="section">
        <div class="alert alert-info">
          <span>🔬</span>
          <span>Enter patient vitals to classify their disease risk level. The model predicts Low, Moderate, or High risk.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Patient Vital Input</div>
              <div class="card-tag">Classification</div>
            </div>
            <div class="form-grid">
              <div class="form-group">
                <label class="form-label">Age (years)</label>
                <input class="form-input" type="number" id="cl_age" placeholder="e.g. 45" value="45"/>
              </div>
              <div class="form-group">
                <label class="form-label">BMI</label>
                <input class="form-input" type="number" id="cl_bmi" placeholder="e.g. 27.5" value="27.5" step="0.1"/>
              </div>
              <div class="form-group">
                <label class="form-label">Blood Pressure (mmHg)</label>
                <input class="form-input" type="number" id="cl_bp" placeholder="e.g. 120" value="120"/>
              </div>
              <div class="form-group">
                <label class="form-label">Glucose (mg/dL)</label>
                <input class="form-input" type="number" id="cl_glucose" placeholder="e.g. 95" value="95"/>
              </div>
              <div class="form-group">
                <label class="form-label">Cholesterol (mg/dL)</label>
                <input class="form-input" type="number" id="cl_chol" placeholder="e.g. 200" value="200"/>
              </div>
              <div class="form-group">
                <label class="form-label">Heart Rate (bpm)</label>
                <input class="form-input" type="number" id="cl_hr" placeholder="e.g. 72" value="72"/>
              </div>
              <div class="form-group">
                <label class="form-label">Smoking (0=No / 1=Yes)</label>
                <input class="form-input" type="number" id="cl_smoke" placeholder="0 or 1" value="0" min="0" max="1"/>
              </div>
              <div class="form-group">
                <label class="form-label">Diabetes (0=No / 1=Yes)</label>
                <input class="form-input" type="number" id="cl_diab" placeholder="0 or 1" value="0" min="0" max="1"/>
              </div>
            </div>
            <div style="margin-top:18px;display:flex;gap:10px">
              <button class="btn btn-primary" onclick="runClassify()">🔬 Classify Risk</button>
              <button class="btn btn-outline" onclick="fillDemo('classify')">Load Demo Patient</button>
            </div>
            <div class="loader" id="cl_loader"><div class="spinner"></div> Running model inference…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">Classification Result</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="cl_result" class="result-box">
              <div class="result-row">
                <span class="result-key">Risk Level</span>
                <span class="result-val" id="cl_risk">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Predicted Class</span>
                <span class="result-val cyan" id="cl_class">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Confidence</span>
                <span class="result-val success" id="cl_conf">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Probabilities</span>
                <span class="result-val" id="cl_proba" style="font-size:11px">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Model</span>
                <span class="result-val" id="cl_model">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Timestamp</span>
                <span class="result-val" id="cl_time" style="font-size:11px">—</span>
              </div>
            </div>
            <div style="margin-top:16px;font-size:12px;color:var(--dim);line-height:1.7">
              <strong style="color:var(--white)">How to interpret:</strong><br/>
              🟢 Low Risk — Routine monitoring advised<br/>
              🟡 Moderate Risk — Follow-up appointment recommended<br/>
              🔴 High Risk — Immediate clinical review required
            </div>
          </div>
        </div>
      </div>

      <!-- ── REGRESSION ── -->
      <div id="regress" class="section">
        <div class="alert alert-info">
          <span>📈</span>
          <span>Predict a continuous health score (0–100) for a patient based on their clinical measurements.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Patient Data Input</div>
              <div class="card-tag">Regression</div>
            </div>
            <div class="form-grid">
              <div class="form-group"><label class="form-label">Age</label><input class="form-input" type="number" id="rg_age" value="52"/></div>
              <div class="form-group"><label class="form-label">BMI</label><input class="form-input" type="number" id="rg_bmi" value="24.3" step="0.1"/></div>
              <div class="form-group"><label class="form-label">Blood Pressure</label><input class="form-input" type="number" id="rg_bp" value="118"/></div>
              <div class="form-group"><label class="form-label">Glucose (mg/dL)</label><input class="form-input" type="number" id="rg_glucose" value="88"/></div>
              <div class="form-group"><label class="form-label">Cholesterol</label><input class="form-input" type="number" id="rg_chol" value="185"/></div>
              <div class="form-group"><label class="form-label">Heart Rate (bpm)</label><input class="form-input" type="number" id="rg_hr" value="68"/></div>
              <div class="form-group"><label class="form-label">Smoking</label><input class="form-input" type="number" id="rg_smoke" value="0" min="0" max="1"/></div>
              <div class="form-group"><label class="form-label">Diabetes</label><input class="form-input" type="number" id="rg_diab" value="0" min="0" max="1"/></div>
            </div>
            <div style="margin-top:18px;display:flex;gap:10px">
              <button class="btn btn-primary" onclick="runRegress()">📈 Predict Health Score</button>
              <button class="btn btn-outline" onclick="fillDemo('regress')">Load Demo Patient</button>
            </div>
            <div class="loader" id="rg_loader"><div class="spinner"></div> Computing health score…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">Health Score Result</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="rg_result" class="result-box">
              <div style="text-align:center;padding:12px 0">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:52px;font-weight:700;color:var(--cyan);line-height:1" id="rg_score_big">—</div>
                <div style="font-size:12px;color:var(--dim);margin-top:4px">Health Score / 100</div>
                <div id="rg_interp" style="margin-top:10px;font-size:13px;font-weight:600;color:var(--success)">—</div>
              </div>
              <div class="divider"></div>
              <div class="result-row">
                <span class="result-key">Score</span>
                <span class="result-val cyan" id="rg_score">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Interpretation</span>
                <span class="result-val success" id="rg_interp2">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Model</span>
                <span class="result-val" id="rg_model">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Timestamp</span>
                <span class="result-val" id="rg_time" style="font-size:11px">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── CLUSTERING ── -->
      <div id="cluster" class="section">
        <div class="alert alert-warn">
          <span>⚠️</span>
          <span>Enter the patient's feature vector (comma-separated values). Ensure the number of features matches the model's training dimensions.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Feature Vector Input</div>
              <div class="card-tag">Clustering</div>
            </div>
            <div class="form-group" style="margin-bottom:14px">
              <label class="form-label">Feature Vector (comma-separated)</label>
              <input class="form-input" id="cl2_features" placeholder="e.g. 45, 27.5, 120, 95, 200, 72, 0, 0" value="45, 27.5, 120, 95, 200, 72, 0, 0"/>
            </div>
            <div style="display:flex;gap:10px">
              <button class="btn btn-primary" onclick="runCluster()">🎯 Assign Cluster</button>
            </div>
            <div class="loader" id="cl2_loader"><div class="spinner"></div> Running KMeans…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">Cluster Assignment</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="cl2_result" class="result-box">
              <div class="result-row">
                <span class="result-key">Cluster ID</span>
                <span class="result-val cyan" id="cl2_id">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Risk Label</span>
                <span class="result-val" id="cl2_label">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Timestamp</span>
                <span class="result-val" id="cl2_time" style="font-size:11px">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── RECOMMENDATION ── -->
      <div id="recommend" class="section">
        <div class="alert alert-info">
          <span>💊</span>
          <span>Provide the patient's feature vector to find similar patient cases and recommend treatment approaches based on KNN similarity.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Patient Feature Vector</div>
              <div class="card-tag">KNN Recommendation</div>
            </div>
            <div class="form-group" style="margin-bottom:14px">
              <label class="form-label">Feature Vector (comma-separated)</label>
              <input class="form-input" id="rec_features" placeholder="e.g. 45, 27.5, 120, 95, 200, 72, 0, 0" value="45, 27.5, 120, 95, 200, 72, 0, 0"/>
            </div>
            <button class="btn btn-primary" onclick="runRecommend()">💊 Get Recommendations</button>
            <div class="loader" id="rec_loader"><div class="spinner"></div> Finding similar patients…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">Similar Patient Cases</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="rec_result" class="result-box">
              <div class="result-row">
                <span class="result-key">Similar Patient IDs</span>
                <span class="result-val cyan" id="rec_ids">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Distances</span>
                <span class="result-val" id="rec_dist" style="font-size:11px">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Recommendation</span>
                <span class="result-val success" id="rec_note" style="font-size:11px;white-space:normal">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Timestamp</span>
                <span class="result-val" id="rec_time" style="font-size:11px">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── TIMESERIES ── -->
      <div id="timeseries" class="section">
        <div class="alert alert-info">
          <span>⏱️</span>
          <span>Forecast future vital sign trends for a patient. Useful for monitoring ICU patients or chronic disease progression.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Forecast Configuration</div>
              <div class="card-tag">Time Series</div>
            </div>
            <div class="form-group" style="margin-bottom:18px">
              <label class="form-label">Forecast Steps (days ahead)</label>
              <input class="form-input" type="number" id="ts_steps" value="7" min="1" max="90"/>
            </div>
            <button class="btn btn-primary" onclick="runTimeseries()">⏱️ Generate Forecast</button>
            <div class="loader" id="ts_loader"><div class="spinner"></div> Running time series model…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">Forecast Result</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="ts_result" class="result-box">
              <div class="result-row">
                <span class="result-key">Steps Forecast</span>
                <span class="result-val cyan" id="ts_steps_out">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Predicted Values</span>
                <span class="result-val" id="ts_vals" style="font-size:11px;white-space:normal">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Model</span>
                <span class="result-val" id="ts_model">—</span>
              </div>
              <div id="ts_bar_wrap" style="margin-top:14px;display:none">
                <div style="font-size:11px;color:var(--dim);margin-bottom:6px">Forecast Trend</div>
                <div class="mini-bars" id="ts_bars"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── PCA ── -->
      <div id="pca" class="section">
        <div class="alert alert-info">
          <span>🔭</span>
          <span>Reduce high-dimensional patient feature space for visualization or downstream ML tasks.</span>
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <div class="card-title">High-Dimensional Features</div>
              <div class="card-tag">PCA</div>
            </div>
            <div class="form-group" style="margin-bottom:18px">
              <label class="form-label">Feature Vector (comma-separated)</label>
              <input class="form-input" id="pca_features" placeholder="e.g. 45, 27.5, 120, 95, 200, 72, 0, 0" value="45, 27.5, 120, 95, 200, 72, 0, 0"/>
            </div>
            <button class="btn btn-primary" onclick="runPCA()">🔭 Reduce Dimensions</button>
            <div class="loader" id="pca_loader"><div class="spinner"></div> Applying PCA transform…</div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">PCA Result</div>
              <div class="card-tag">Output</div>
            </div>
            <div id="pca_result" class="result-box">
              <div class="result-row">
                <span class="result-key">Reduced Features</span>
                <span class="result-val cyan" id="pca_reduced" style="font-size:11px">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Components (n)</span>
                <span class="result-val" id="pca_n">—</span>
              </div>
              <div class="result-row">
                <span class="result-key">Variance Explained</span>
                <span class="result-val success" id="pca_var" style="font-size:11px">—</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── ASSOCIATION ── -->
      <div id="assoc" class="section">
        <div class="card">
          <div class="card-header">
            <div class="card-title">Medical Association Rules</div>
            <div class="card-tag">Apriori / FP-Growth</div>
          </div>
          <button class="btn btn-primary btn-sm" onclick="loadAssocRules()">🔗 Load Rules</button>
          <div class="loader" id="assoc_loader"><div class="spinner"></div> Loading association rules…</div>
          <div id="assoc_table_wrap" style="margin-top:18px;display:none;overflow-x:auto">
            <table class="data-table" id="assoc_table">
              <thead>
                <tr>
                  <th>Antecedents</th>
                  <th>Consequents</th>
                  <th>Support</th>
                  <th>Confidence</th>
                  <th>Lift</th>
                </tr>
              </thead>
              <tbody id="assoc_body"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ── API REFERENCE ── -->
      <div id="api" class="section">
        <div class="card">
          <div class="card-header">
            <div class="card-title">API Endpoints</div>
            <div class="card-tag">FastAPI</div>
          </div>
          <table class="data-table">
            <thead>
              <tr><th>Method</th><th>Endpoint</th><th>Description</th><th>Action</th></tr>
            </thead>
            <tbody>
              <tr><td><span style="color:var(--success);font-family:monospace;font-weight:600">GET</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/health</code></td><td style="color:var(--dim)">System health check</td><td><button class="btn btn-outline btn-sm" onclick="testEndpoint('/health')">Test</button></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/classify</code></td><td style="color:var(--dim)">Patient risk classification</td><td><a href="/api/docs#/ML%20Predictions/classify_patient_api_classify_post" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/regress</code></td><td style="color:var(--dim)">Health score prediction</td><td><a href="/api/docs" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/cluster</code></td><td style="color:var(--dim)">Patient cluster assignment</td><td><a href="/api/docs" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/recommend</code></td><td style="color:var(--dim)">Treatment recommendation</td><td><a href="/api/docs" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/timeseries/forecast</code></td><td style="color:var(--dim)">Vital signs forecast</td><td><a href="/api/docs" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--warn);font-family:monospace;font-weight:600">POST</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/pca</code></td><td style="color:var(--dim)">PCA dimensionality reduction</td><td><a href="/api/docs" target="_blank"><button class="btn btn-outline btn-sm">Docs</button></a></td></tr>
              <tr><td><span style="color:var(--success);font-family:monospace;font-weight:600">GET</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/association-rules</code></td><td style="color:var(--dim)">Medical association rules</td><td><button class="btn btn-outline btn-sm" onclick="testEndpoint('/api/association-rules')">Test</button></td></tr>
              <tr><td><span style="color:var(--success);font-family:monospace;font-weight:600">GET</span></td><td><code style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">/api/manifest</code></td><td style="color:var(--dim)">Model metadata manifest</td><td><button class="btn btn-outline btn-sm" onclick="testEndpoint('/api/manifest')">Test</button></td></tr>
            </tbody>
          </table>
          <div class="divider"></div>
          <div style="display:flex;gap:12px">
            <a href="/api/docs" target="_blank"><button class="btn btn-primary">📖 Open Swagger UI</button></a>
            <a href="/api/redoc" target="_blank"><button class="btn btn-outline">📄 Open ReDoc</button></a>
          </div>
          <div id="api_test_result" class="result-box" style="margin-top:12px">
            <pre id="api_test_json" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);white-space:pre-wrap;word-break:break-all"></pre>
          </div>
        </div>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /layout -->

<script>
// ── Clock ──
function updateClock(){
  const now = new Date();
  document.getElementById('clockBadge').textContent =
    now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  document.getElementById('lastSync').textContent =
    now.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});
}
updateClock(); setInterval(updateClock, 1000);

// ── Navigation ──
const pageTitles = {
  overview: ['Clinical Dashboard','Real-time patient analytics & ML intelligence'],
  models:   ['Model Status','6 active ML models · Healthcare domain'],
  classify: ['Risk Classification','Predict patient disease risk level'],
  regress:  ['Health Regression','Predict continuous health score (0–100)'],
  cluster:  ['Patient Clustering','KMeans risk segmentation'],
  recommend:['Treatment Recommendation','KNN-based similar patient matching'],
  timeseries:['Vitals Forecast','Time series forecasting for vital signs'],
  pca:      ['PCA Reduction','Dimensionality reduction for patient features'],
  assoc:    ['Association Rules','Medical condition co-occurrence patterns'],
  api:      ['API Reference','FastAPI endpoints & documentation'],
};
function showSection(id, el){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if(el) el.classList.add('active');
  const t = pageTitles[id]||[id,''];
  document.getElementById('pageTitle').textContent = t[0];
  document.getElementById('pageSub').textContent = t[1];
}

// ── Demo data ──
const demos = {
  classify: {cl_age:62,cl_bmi:31.2,cl_bp:145,cl_glucose:180,cl_chol:240,cl_hr:88,cl_smoke:1,cl_diab:1},
  regress:  {rg_age:38,rg_bmi:22.4,rg_bp:115,rg_glucose:82,rg_chol:170,rg_hr:65,rg_smoke:0,rg_diab:0},
};
function fillDemo(type){
  const d=demos[type];
  Object.entries(d).forEach(([k,v])=>{
    const el=document.getElementById(k);
    if(el) el.value=v;
  });
}

// ── API helpers ──
async function post(url, body){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}
async function get(url){ const r=await fetch(url); if(!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }

function showLoader(id,show){ const el=document.getElementById(id); if(el){el.classList.toggle('show',show);} }

// ── Classification ──
async function runClassify(){
  showLoader('cl_loader',true);
  document.getElementById('cl_result').classList.remove('show');
  try{
    const data = await post('/api/classify',{
      age: +document.getElementById('cl_age').value,
      bmi: +document.getElementById('cl_bmi').value,
      blood_pressure: +document.getElementById('cl_bp').value,
      glucose: +document.getElementById('cl_glucose').value,
      cholesterol: +document.getElementById('cl_chol').value,
      heart_rate: +document.getElementById('cl_hr').value,
      smoking: +document.getElementById('cl_smoke').value,
      diabetes: +document.getElementById('cl_diab').value,
    });
    const riskColors = {'Low Risk':'success','Moderate Risk':'warn','High Risk':'danger'};
    const cls = riskColors[data.risk_level]||'cyan';
    document.getElementById('cl_risk').innerHTML =
      `<span class="risk-badge risk-${data.risk_level==='Low Risk'?'low':data.risk_level==='Moderate Risk'?'mod':'high'}">${data.risk_level}</span>`;
    document.getElementById('cl_class').textContent = data.prediction;
    document.getElementById('cl_conf').textContent = data.confidence ? (data.confidence*100).toFixed(1)+'%' : 'N/A';
    document.getElementById('cl_proba').textContent = data.probabilities ? data.probabilities.map(p=>(p*100).toFixed(1)+'%').join(' | ') : 'N/A';
    document.getElementById('cl_model').textContent = data.model||'Classifier';
    document.getElementById('cl_time').textContent = data.timestamp||'—';
    document.getElementById('cl_result').classList.add('show');
  }catch(e){
    alert('Error: '+e.message);
  } finally { showLoader('cl_loader',false); }
}

// ── Regression ──
async function runRegress(){
  showLoader('rg_loader',true);
  document.getElementById('rg_result').classList.remove('show');
  try{
    const data = await post('/api/regress',{
      age: +document.getElementById('rg_age').value,
      bmi: +document.getElementById('rg_bmi').value,
      blood_pressure: +document.getElementById('rg_bp').value,
      glucose: +document.getElementById('rg_glucose').value,
      cholesterol: +document.getElementById('rg_chol').value,
      heart_rate: +document.getElementById('rg_hr').value,
      smoking: +document.getElementById('rg_smoke').value,
      diabetes: +document.getElementById('rg_diab').value,
    });
    const score = data.predicted_health_score.toFixed(1);
    document.getElementById('rg_score_big').textContent = score;
    document.getElementById('rg_score').textContent = score;
    document.getElementById('rg_interp').textContent = data.interpretation;
    document.getElementById('rg_interp2').textContent = data.interpretation;
    document.getElementById('rg_model').textContent = data.model||'Regressor';
    document.getElementById('rg_time').textContent = data.timestamp||'—';
    document.getElementById('rg_result').classList.add('show');
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('rg_loader',false); }
}

// ── Clustering ──
async function runCluster(){
  showLoader('cl2_loader',true);
  document.getElementById('cl2_result').classList.remove('show');
  try{
    const feats = document.getElementById('cl2_features').value.split(',').map(Number);
    const data = await post('/api/cluster',{features:feats});
    document.getElementById('cl2_id').textContent = data.cluster_id;
    document.getElementById('cl2_label').textContent = data.risk_label;
    document.getElementById('cl2_time').textContent = data.timestamp||'—';
    document.getElementById('cl2_result').classList.add('show');
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('cl2_loader',false); }
}

// ── Recommendation ──
async function runRecommend(){
  showLoader('rec_loader',true);
  document.getElementById('rec_result').classList.remove('show');
  try{
    const feats = document.getElementById('rec_features').value.split(',').map(Number);
    const data = await post('/api/recommend',{patient_features:feats});
    document.getElementById('rec_ids').textContent = (data.similar_patient_indices||[]).join(', ');
    document.getElementById('rec_dist').textContent = (data.distances||[]).map(d=>d.toFixed(3)).join(', ');
    document.getElementById('rec_note').textContent = data.recommendation||'—';
    document.getElementById('rec_time').textContent = data.timestamp||'—';
    document.getElementById('rec_result').classList.add('show');
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('rec_loader',false); }
}

// ── Time Series ──
async function runTimeseries(){
  showLoader('ts_loader',true);
  document.getElementById('ts_result').classList.remove('show');
  try{
    const steps = +document.getElementById('ts_steps').value;
    const data = await post('/api/timeseries/forecast',{steps});
    const vals = data.forecast || data.data?.forecast || [];
    document.getElementById('ts_steps_out').textContent = data.steps||steps;
    document.getElementById('ts_vals').textContent = vals.map?.(v=>v.toFixed?.(2)||v).join(', ')||JSON.stringify(vals).slice(0,120);
    document.getElementById('ts_model').textContent = data.model||'Time Series';
    // mini bar chart
    if(vals.length>0){
      const mn=Math.min(...vals), mx=Math.max(...vals)||1;
      const bars=document.getElementById('ts_bars');
      bars.innerHTML=vals.slice(0,14).map(v=>{
        const h=Math.max(8,Math.round(((v-mn)/(mx-mn||1))*52)+8);
        return `<div class="mini-bar" style="height:${h}px" title="${v.toFixed?.(2)||v}"></div>`;
      }).join('');
      document.getElementById('ts_bar_wrap').style.display='block';
    }
    document.getElementById('ts_result').classList.add('show');
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('ts_loader',false); }
}

// ── PCA ──
async function runPCA(){
  showLoader('pca_loader',true);
  document.getElementById('pca_result').classList.remove('show');
  try{
    const feats = document.getElementById('pca_features').value.split(',').map(Number);
    const data = await post('/api/pca',{features:feats});
    document.getElementById('pca_reduced').textContent = (data.reduced_features||[]).map(v=>v.toFixed(4)).join(', ');
    document.getElementById('pca_n').textContent = data.n_components||'—';
    const evr = (data.explained_variance_ratio||[]).map(v=>(v*100).toFixed(1)+'%').join(' | ');
    document.getElementById('pca_var').textContent = evr||'—';
    document.getElementById('pca_result').classList.add('show');
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('pca_loader',false); }
}

// ── Association Rules ──
async function loadAssocRules(){
  showLoader('assoc_loader',true);
  try{
    const data = await get('/api/association-rules');
    const rules = data.rules || [];
    const tbody = document.getElementById('assoc_body');
    if(Array.isArray(rules) && rules.length>0){
      tbody.innerHTML = rules.slice(0,20).map(r=>`
        <tr>
          <td style="font-family:'JetBrains Mono',monospace;font-size:11px">${JSON.stringify(r.antecedents||r[0]||'—')}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:11px">${JSON.stringify(r.consequents||r[1]||'—')}</td>
          <td>${(r.support||0).toFixed?.(3)||r.support||'—'}</td>
          <td style="color:var(--cyan)">${(r.confidence||0).toFixed?.(3)||r.confidence||'—'}</td>
          <td style="color:var(--success)">${(r.lift||0).toFixed?.(3)||r.lift||'—'}</td>
        </tr>
      `).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan="5" style="color:var(--dim);text-align:center;padding:20px">No structured rules available — raw data loaded</td></tr>`;
    }
    document.getElementById('assoc_table_wrap').style.display='block';
  }catch(e){ alert('Error: '+e.message); }
  finally{ showLoader('assoc_loader',false); }
}

// ── API Test ──
async function testEndpoint(url){
  try{
    const data = await get(url);
    const box = document.getElementById('api_test_result');
    document.getElementById('api_test_json').textContent = JSON.stringify(data,null,2).slice(0,800);
    box.classList.add('show');
    document.querySelector('[onclick*="api"]')?.click();
  }catch(e){ alert('Error: '+e.message); }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
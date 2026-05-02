"""
VitalWatch - Sepsis Early Warning & ICU Intelligence Platform
FastAPI Backend + Hospital-Grade Dashboard
AI221 Machine Learning Semester Project
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import numpy as np
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VitalWatch")

# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────
app = FastAPI(
    title="VitalWatch Healthcare ML API",
    description="Sepsis Early Warning & ICU Patient Deterioration System",
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
#  Feature columns — must match training order
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
    'age', 'gender',
    'hr_mean_3h', 'hr_mean_6h', 'sbp_mean_3h', 'map_mean_3h', 'resp_mean_3h',
    'hr_trend', 'sbp_trend', 'temp_trend', 'resp_trend',
    'hr_std_3h', 'sbp_std_3h',
    'flag_tachy', 'flag_hypoxia', 'flag_fever', 'flag_hypotemp',
    'flag_low_bp', 'flag_tachypnea', 'qsofa', 'hours_in_icu'
]

# ─────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_pkl(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        logger.warning(f"Not found: {path}")
        return None
    return joblib.load(path)

def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

logger.info("Loading VitalWatch models...")
classifier       = load_pkl("best_classifier.pkl")
regressor        = load_pkl("best_regressor.pkl")
timeseries_model = load_pkl("best_timeseries.pkl")
kmeans           = load_pkl("kmeans_clustering.pkl")
knn_recommender  = load_pkl("knn_recommendation.pkl")
pca_reducer      = load_pkl("pca_reducer.pkl")
scaler           = load_pkl("scaler.pkl")
rec_scaler       = load_pkl("recommendation_scaler.pkl")
risk_labels      = load_pkl("cluster_risk_labels.pkl")
patient_profiles = load_pkl("patient_profiles.pkl")

manifest         = load_json("manifest.json")
assoc_rules      = load_json("association_rules.json")
cluster_analysis = load_json("cluster_analysis.json")
pca_results_json = load_json("pca_results.json")
ts_results       = load_json("timeseries_results.json")

logger.info("All models loaded.")

# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────
class PatientVitals(BaseModel):
    hr:            float = Field(85.0,  description="Heart rate (bpm)")
    o2sat:         float = Field(96.0,  description="O2 saturation (%)")
    temp:          float = Field(37.0,  description="Temperature (°C)")
    sbp:           float = Field(120.0, description="Systolic BP (mmHg)")
    map:           float = Field(80.0,  description="Mean arterial pressure")
    dbp:           float = Field(70.0,  description="Diastolic BP")
    resp:          float = Field(16.0,  description="Respiratory rate")
    age:           float = Field(50.0,  description="Age (years)")
    gender:        float = Field(1.0,   description="Gender 0=F 1=M")
    hr_mean_3h:    float = Field(85.0)
    hr_mean_6h:    float = Field(85.0)
    sbp_mean_3h:   float = Field(120.0)
    map_mean_3h:   float = Field(80.0)
    resp_mean_3h:  float = Field(16.0)
    hr_trend:      float = Field(0.0)
    sbp_trend:     float = Field(0.0)
    temp_trend:    float = Field(0.0)
    resp_trend:    float = Field(0.0)
    hr_std_3h:     float = Field(2.0)
    sbp_std_3h:    float = Field(3.0)
    flag_tachy:    float = Field(0.0)
    flag_hypoxia:  float = Field(0.0)
    flag_fever:    float = Field(0.0)
    flag_hypotemp: float = Field(0.0)
    flag_low_bp:   float = Field(0.0)
    flag_tachypnea:float = Field(0.0)
    qsofa:         float = Field(0.0)
    hours_in_icu:  float = Field(1.0)

class TimeSeriesInput(BaseModel):
    hr_last_6_hours: List[float]
    patient_id: Optional[str] = "unknown"

class RecommendationInput(BaseModel):
    avg_hr:          float = 85.0
    avg_temp:        float = 37.0
    avg_o2sat:       float = 96.0
    avg_sbp:         float = 120.0
    avg_map:         float = 80.0
    avg_resp:        float = 16.0
    max_qsofa:       float = 0.0
    had_tachy:       float = 0.0
    had_fever:       float = 0.0
    had_low_bp:      float = 0.0
    had_hypoxia:     float = 0.0
    total_icu_hours: float = 24.0

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def prepare(vitals: PatientVitals) -> np.ndarray:
    vals = np.array([[getattr(vitals, c) for c in FEATURE_COLS]])
    return scaler.transform(vals) if scaler else vals

def risk_level(prob: float) -> str:
    if prob >= 0.7:  return "HIGH RISK"
    if prob >= 0.4:  return "MODERATE RISK"
    return "LOW RISK"

def clinical_action(level: str) -> str:
    return {
        "HIGH RISK":     "Initiate sepsis protocol immediately. Order blood cultures & lactate. Start IV antibiotics within 1 hour.",
        "MODERATE RISK": "Increase monitoring to every 30 minutes. Order CBC and metabolic panel. Notify attending physician.",
        "LOW RISK":      "Continue standard monitoring every 4 hours. No immediate intervention required."
    }.get(level, "Follow standard protocol")

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "operational",
        "service": "VitalWatch",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "classifier":  classifier  is not None,
            "regressor":   regressor   is not None,
            "timeseries":  timeseries_model is not None,
            "clustering":  kmeans      is not None,
            "recommender": knn_recommender is not None,
            "pca":         pca_reducer is not None,
        }
    }

@app.get("/api/manifest", tags=["System"])
async def get_manifest():
    return manifest or {"message": "No manifest available"}

# ── 1. Full Assessment (main dashboard endpoint) ──
@app.post("/api/assess", tags=["ML Predictions"])
async def full_assessment(patient: PatientVitals):
    try:
        X = prepare(patient)

        prob       = float(classifier.predict_proba(X)[0][1]) if classifier else 0.5
        level      = risk_level(prob)
        next_hr    = float(regressor.predict(X)[0]) if regressor else patient.hr
        cluster_id = int(kmeans.predict(X)[0]) if kmeans else 0
        c_name     = risk_labels.get(cluster_id, "Unknown") if risk_labels else f"Cluster {cluster_id}"
        pca_coords = pca_reducer.transform(X)[0] if pca_reducer else [0, 0]

        flags = []
        if patient.flag_tachy:     flags.append("Tachycardia")
        if patient.flag_fever:     flags.append("Fever")
        if patient.flag_hypoxia:   flags.append("Hypoxia")
        if patient.flag_low_bp:    flags.append("Low Blood Pressure")
        if patient.flag_tachypnea: flags.append("Tachypnea")
        if patient.qsofa >= 2:     flags.append("High qSOFA")

        return {
            "sepsis": {
                "probability": round(prob, 4),
                "risk_level":  level,
                "action":      clinical_action(level),
                "flags":       flags,
            },
            "hr_forecast": {
                "current":   patient.hr,
                "next_hour": round(next_hr, 1),
                "trend":     "rising" if next_hr > patient.hr + 2 else "falling" if next_hr < patient.hr - 2 else "stable"
            },
            "cluster": {
                "id":   cluster_id,
                "name": c_name,
            },
            "pca": {
                "pc1": round(float(pca_coords[0]), 4),
                "pc2": round(float(pca_coords[1]), 4),
            },
            "vitals": {
                "hr": patient.hr, "temp": patient.temp,
                "o2sat": patient.o2sat, "map": patient.map,
                "sbp": patient.sbp, "resp": patient.resp,
                "hours_in_icu": patient.hours_in_icu,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 2. Sepsis Classification ──
@app.post("/api/classify", tags=["ML Predictions"])
async def classify(patient: PatientVitals):
    if not classifier:
        raise HTTPException(503, "Classifier not loaded")
    try:
        X     = prepare(patient)
        pred  = int(classifier.predict(X)[0])
        prob  = float(classifier.predict_proba(X)[0][1])
        level = risk_level(prob)
        return {
            "prediction":  pred,
            "probability": round(prob, 4),
            "risk_level":  level,
            "action":      clinical_action(level),
            "timestamp":   datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 3. HR Regression ──
@app.post("/api/regress", tags=["ML Predictions"])
async def regress(patient: PatientVitals):
    if not regressor:
        raise HTTPException(503, "Regressor not loaded")
    try:
        X        = prepare(patient)
        next_hr  = float(regressor.predict(X)[0])
        trend    = next_hr - patient.hr
        return {
            "current_hr":    patient.hr,
            "predicted_hr":  round(next_hr, 1),
            "change":        round(trend, 1),
            "trend":         "rising" if trend > 2 else "falling" if trend < -2 else "stable",
            "status":        "CRITICAL" if next_hr > 120 else "WARNING" if next_hr > 100 else "NORMAL",
            "timestamp":     datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 4. Clustering ──
@app.post("/api/cluster", tags=["ML Predictions"])
async def cluster(patient: PatientVitals):
    if not kmeans:
        raise HTTPException(503, "Clustering model not loaded")
    try:
        X          = prepare(patient)
        cluster_id = int(kmeans.predict(X)[0])
        distances  = kmeans.transform(X)[0]
        label      = risk_labels.get(cluster_id, f"Cluster {cluster_id}") if risk_labels else f"Cluster {cluster_id}"
        return {
            "cluster_id":   cluster_id,
            "risk_label":   label,
            "all_distances": {f"cluster_{i}": round(float(d), 3) for i, d in enumerate(distances)},
            "timestamp":    datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 5. PCA ──
@app.post("/api/pca", tags=["ML Predictions"])
async def pca(patient: PatientVitals):
    if not pca_reducer:
        if pca_results_json:
            return {"source": "cached", "data": pca_results_json}
        raise HTTPException(503, "PCA not loaded")
    try:
        X      = prepare(patient)
        coords = pca_reducer.transform(X)[0]
        pc1, pc2 = float(coords[0]), float(coords[1])
        zone   = "RED ZONE" if pc1 > 1.5 or pc2 > 1.5 else "YELLOW ZONE" if pc1 > 0.5 or pc2 > 0.5 else "GREEN ZONE"
        return {
            "pc1":  round(pc1, 4),
            "pc2":  round(pc2, 4),
            "zone": zone,
            "explained_variance": pca_reducer.explained_variance_ratio_.tolist() if hasattr(pca_reducer, 'explained_variance_ratio_') else [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 6. Time Series Forecast ──
@app.post("/api/forecast", tags=["ML Predictions"])
async def forecast(data: TimeSeriesInput):
    try:
        if len(data.hr_last_6_hours) != 6:
            raise HTTPException(400, f"Need exactly 6 HR values. Got {len(data.hr_last_6_hours)}")

        window = np.array(data.hr_last_6_hours)

        if not timeseries_model:
            # Simple linear extrapolation fallback
            trend   = (window[-1] - window[0]) / 5
            forecast_vals = [round(window[-1] + trend * (i+1), 1) for i in range(6)]
        else:
            forecast_vals = []
            cur = list(window)
            for _ in range(6):
                w    = np.array(cur[-6:])
                feat = np.array([[*w, w.mean(), w.std(), w[-1]-w[0]]])
                p    = float(timeseries_model.predict(feat)[0])
                forecast_vals.append(round(p, 1))
                cur.append(p)

        result = [{"hour": f"+{i+1}h", "hr": v, "alert": v > 100} for i, v in enumerate(forecast_vals)]
        danger = [r for r in result if r["alert"]]

        return {
            "patient_id":      data.patient_id,
            "input_hrs":       data.hr_last_6_hours,
            "forecast":        result,
            "danger_hours":    danger,
            "alert":           len(danger) > 0,
            "alert_message":   f"HR predicted >100bpm in {len(danger)} of next 6 hours" if danger else "No danger predicted",
            "timestamp":       datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 7. Recommendation ──
@app.post("/api/recommend", tags=["ML Predictions"])
async def recommend(profile: RecommendationInput):
    if not knn_recommender:
        if ts_results:
            return {"source": "cached", "data": ts_results}
        raise HTTPException(503, "Recommender not loaded")
    try:
        REC_FEATURES = ['avg_hr','avg_temp','avg_o2sat','avg_sbp','avg_map',
                        'avg_resp','max_qsofa','had_tachy','had_fever',
                        'had_low_bp','had_hypoxia','total_icu_hours']
        vals   = np.array([[getattr(profile, f) for f in REC_FEATURES]])
        scaled = rec_scaler.transform(vals) if rec_scaler else vals
        distances, indices = knn_recommender.kneighbors(scaled)

        similar = []
        if patient_profiles is not None:
            sim_rows = patient_profiles.iloc[indices[0][1:]]
            for i, (idx, row) in enumerate(sim_rows.iterrows()):
                sim = round(1 / (1 + distances[0][i+1]) * 100, 1)
                similar.append({
                    "rank": i+1,
                    "patient_id": str(row.get('patient_id', idx)),
                    "had_sepsis": bool(row.get('had_sepsis', False)),
                    "similarity": f"{sim}%"
                })
            sepsis_count = int(sim_rows['had_sepsis'].sum()) if 'had_sepsis' in sim_rows else 0
        else:
            sepsis_count = 0
            similar = [{"rank": i+1, "patient_id": str(idx), "similarity": f"{round(1/(1+distances[0][i+1])*100,1)}%"} for i, idx in enumerate(indices[0][1:])]

        risk_pct = round(sepsis_count / max(len(similar), 1) * 100)
        level    = "HIGH RISK" if sepsis_count >= 5 else "MODERATE RISK" if sepsis_count >= 2 else "LOW RISK"
        return {
            "similar_patients":    similar,
            "sepsis_in_similar":   sepsis_count,
            "estimated_risk_pct":  risk_pct,
            "risk_level":          level,
            "recommendation":      clinical_action(level),
            "timestamp":           datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

# ── 8. Association Rules ──
@app.get("/api/association-rules", tags=["Analytics"])
async def association_rules():
    if not assoc_rules:
        raise HTTPException(503, "Association rules not available")
    return {
        "total_rules":    assoc_rules.get("total_rules", 0),
        "sepsis_rules":   assoc_rules.get("sepsis_rules_count", 0),
        "top_rules":      assoc_rules.get("top_10_by_lift", []),
        "note":           assoc_rules.get("note", ""),
    }


# ═══════════════════════════════════════════════════════════════
#  HOSPITAL-GRADE DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VitalWatch — Sepsis Early Warning System</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --navy:   #060e1c;
    --navy2:  #0c1a30;
    --blue:   #1a4fa0;
    --cyan:   #00c8f0;
    --teal:   #0096c7;
    --white:  #e8f4ff;
    --dim:    #7a96b8;
    --ok:     #2ec4b6;
    --warn:   #f4a261;
    --danger: #e63946;
    --card:   rgba(12,26,48,0.95);
    --border: rgba(0,200,240,0.14);
    --glow:   0 0 28px rgba(0,200,240,0.1);
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{font-family:'DM Sans',sans-serif;background:var(--navy);color:var(--white);min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;z-index:0;
    background:radial-gradient(ellipse 60% 50% at 15% 10%,rgba(0,100,200,0.16) 0%,transparent 70%),
               radial-gradient(ellipse 50% 40% at 85% 85%,rgba(0,200,240,0.1) 0%,transparent 65%),
               repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(0,200,240,0.03) 39px,rgba(0,200,240,0.03) 40px),
               repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(0,200,240,0.03) 39px,rgba(0,200,240,0.03) 40px);
    pointer-events:none}

  .layout{display:flex;min-height:100vh;position:relative;z-index:1}

  /* SIDEBAR */
  .sidebar{width:250px;flex-shrink:0;background:rgba(6,14,28,.97);border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;backdrop-filter:blur(16px)}
  .sb-logo{padding:24px 20px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
  .sb-icon{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,var(--blue),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 18px rgba(0,200,240,0.35)}
  .sb-name{font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;color:var(--white);line-height:1.2}
  .sb-sub{font-size:10px;color:var(--dim);letter-spacing:.4px}
  .sb-nav{flex:1;padding:14px 10px;overflow-y:auto}
  .nav-sec{margin-bottom:20px}
  .nav-lbl{font-size:9.5px;font-weight:700;letter-spacing:1.5px;color:var(--dim);text-transform:uppercase;padding:0 10px;margin-bottom:4px}
  .nav-item{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:7px;cursor:pointer;transition:all .2s;font-size:13px;font-weight:500;color:var(--dim);border:1px solid transparent;margin-bottom:1px}
  .nav-item:hover{background:rgba(0,200,240,0.07);color:var(--white);border-color:var(--border)}
  .nav-item.active{background:linear-gradient(135deg,rgba(0,80,180,0.28),rgba(0,200,240,0.14));color:var(--cyan);border-color:rgba(0,200,240,0.28);box-shadow:var(--glow)}
  .nav-item .ic{font-size:15px;width:18px;text-align:center}
  .sb-foot{padding:14px 18px;border-top:1px solid var(--border);font-size:11px;color:var(--dim);line-height:1.7}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 6px var(--ok);margin-right:5px;animation:blink 2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}

  /* MAIN */
  .main{flex:1;display:flex;flex-direction:column;overflow:hidden}
  .topbar{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;background:rgba(6,14,28,.8);border-bottom:1px solid var(--border);backdrop-filter:blur(12px);position:sticky;top:0;z-index:50}
  .pg-title{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;letter-spacing:-.4px}
  .pg-sub{font-size:11.5px;color:var(--dim);margin-top:1px}
  .tb-right{display:flex;align-items:center;gap:10px}
  .clock{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan);background:rgba(0,200,240,0.07);border:1px solid var(--border);padding:5px 12px;border-radius:6px}
  .avatar{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;border:2px solid rgba(0,200,240,0.3)}
  .content{flex:1;padding:24px 28px;overflow-y:auto}

  /* SECTIONS */
  .sec{display:none}
  .sec.active{display:block;animation:fadeUp .3s ease}
  @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

  /* STAT GRID */
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
  .stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden;cursor:default;transition:transform .2s,box-shadow .2s}
  .stat:hover{transform:translateY(-2px);box-shadow:var(--glow)}
  .stat::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--blue),var(--cyan))}
  .stat.warn::before{background:linear-gradient(90deg,#f4a261,#e76f51)}
  .stat.danger::before{background:linear-gradient(90deg,#e63946,#c1121f)}
  .stat.ok::before{background:linear-gradient(90deg,var(--ok),#06d6a0)}
  .stat-ico{font-size:26px;margin-bottom:8px}
  .stat-val{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;letter-spacing:-1px;line-height:1}
  .stat-lbl{font-size:11.5px;color:var(--dim);margin-top:3px}
  .stat-note{font-size:10.5px;margin-top:7px;color:var(--ok)}

  /* CARD */
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:22px;margin-bottom:18px;box-shadow:0 4px 24px rgba(0,0,0,.2)}
  .card-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
  .card-title{font-family:'Space Grotesk',sans-serif;font-size:14.5px;font-weight:600}
  .card-tag{font-size:9.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;padding:3px 9px;border-radius:99px;background:rgba(0,200,240,0.1);color:var(--cyan);border:1px solid rgba(0,200,240,0.22)}
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}

  /* FORM */
  .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .fg{display:flex;flex-direction:column;gap:5px}
  .fl{font-size:11.5px;font-weight:500;color:var(--dim)}
  .fi{background:rgba(6,14,28,.9);border:1px solid var(--border);border-radius:7px;padding:9px 12px;color:var(--white);font-size:13px;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .2s,box-shadow .2s}
  .fi:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(0,200,240,0.1)}
  .fi::placeholder{color:rgba(122,150,184,.45)}

  /* BUTTONS */
  .btn{padding:9px 20px;border-radius:7px;border:none;font-family:'DM Sans',sans-serif;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:7px}
  .btn-p{background:linear-gradient(135deg,var(--blue),var(--teal));color:#fff;box-shadow:0 4px 14px rgba(0,100,200,.28)}
  .btn-p:hover{transform:translateY(-1px);box-shadow:0 8px 22px rgba(0,100,200,.38)}
  .btn-o{background:transparent;color:var(--cyan);border:1px solid var(--border)}
  .btn-o:hover{background:rgba(0,200,240,.07);border-color:var(--cyan)}
  .btn-sm{padding:6px 14px;font-size:12px}

  /* RESULT */
  .result{background:rgba(0,10,30,.6);border:1px solid var(--border);border-radius:9px;padding:16px;margin-top:14px;display:none}
  .result.show{display:block;animation:fadeUp .3s ease}
  .rrow{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(0,200,240,.06)}
  .rrow:last-child{border-bottom:none}
  .rk{font-size:11.5px;color:var(--dim)}
  .rv{font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace}
  .rv.c{color:var(--cyan)}
  .rv.g{color:var(--ok)}
  .rv.w{color:var(--warn)}
  .rv.d{color:var(--danger)}

  /* BADGE */
  .badge{display:inline-flex;align-items:center;gap:5px;padding:3px 12px;border-radius:99px;font-size:11.5px;font-weight:700;letter-spacing:.4px}
  .badge-low{background:rgba(46,196,182,.14);color:var(--ok);border:1px solid rgba(46,196,182,.28)}
  .badge-mod{background:rgba(244,162,97,.14);color:var(--warn);border:1px solid rgba(244,162,97,.28)}
  .badge-high{background:rgba(230,57,70,.14);color:var(--danger);border:1px solid rgba(230,57,70,.28)}

  /* ACTION BOX */
  .action-box{border-left:3px solid var(--cyan);padding:10px 14px;background:rgba(0,200,240,.05);font-size:12.5px;line-height:1.6;color:var(--white);margin-top:12px;border-radius:0 7px 7px 0}
  .action-box.mod{border-color:var(--warn);background:rgba(244,162,97,.05)}
  .action-box.high{border-color:var(--danger);background:rgba(230,57,70,.05)}

  /* FLAGS */
  .flags{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
  .flag{font-size:11px;padding:3px 9px;border-radius:99px;background:rgba(230,57,70,.12);color:var(--danger);border:1px solid rgba(230,57,70,.25)}

  /* LOADER */
  .loader{display:none;align-items:center;gap:9px;font-size:12.5px;color:var(--cyan);margin-top:10px}
  .loader.show{display:flex}
  .spin{width:15px;height:15px;border-radius:50%;border:2px solid rgba(0,200,240,.18);border-top-color:var(--cyan);animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* DIVIDER */
  .div{height:1px;background:var(--border);margin:14px 0}

  /* FORECAST GRID */
  .fc-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-top:12px}
  .fc-cell{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:6px;padding:8px 4px;text-align:center}
  .fc-cell.alert{border-color:var(--danger);background:rgba(230,57,70,.07)}
  .fc-h{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim)}
  .fc-v{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;color:var(--cyan)}
  .fc-v.hot{color:var(--danger)}

  /* TABLE */
  .tbl{width:100%;border-collapse:collapse;font-size:12.5px}
  .tbl th{text-align:left;padding:9px 12px;font-size:10.5px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--border)}
  .tbl td{padding:9px 12px;border-bottom:1px solid rgba(0,200,240,.04);color:var(--white)}
  .tbl tr:hover td{background:rgba(0,200,240,.03)}

  /* MODEL CARD */
  .mc{background:rgba(6,14,28,.85);border:1px solid var(--border);border-radius:9px;padding:15px}
  .mc-name{font-size:13px;font-weight:600;margin-bottom:2px}
  .mc-type{font-size:10.5px;color:var(--cyan);margin-bottom:10px}
  .mmet{display:flex;justify-content:space-between;margin-bottom:5px}
  .ml{font-size:10.5px;color:var(--dim)}
  .mv{font-size:10.5px;font-weight:600;font-family:'JetBrains Mono',monospace}
  .mbar{height:4px;background:rgba(0,200,240,.08);border-radius:99px;margin-bottom:9px;overflow:hidden}
  .mfill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--cyan))}

  /* ALERT */
  .info-bar{display:flex;align-items:center;gap:10px;padding:11px 15px;border-radius:9px;margin-bottom:16px;font-size:12.5px;background:rgba(0,200,240,.08);border:1px solid rgba(0,200,240,.2);color:var(--cyan)}
  .warn-bar{background:rgba(244,162,97,.08);border:1px solid rgba(244,162,97,.2);color:var(--warn)}

  ::-webkit-scrollbar{width:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:rgba(0,200,240,.18);border-radius:99px}
  @media(max-width:1100px){.stat-grid{grid-template-columns:1fr 1fr}}
  @media(max-width:850px){.two-col,.three-col{grid-template-columns:1fr}}
  @media(max-width:700px){.sidebar{display:none}.content{padding:14px}}
</style>
</head>
<body>
<div class="layout">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sb-logo">
    <div class="sb-icon">❤️</div>
    <div>
      <div class="sb-name">VitalWatch</div>
      <div class="sb-sub">Sepsis Early Warning System</div>
    </div>
  </div>
  <nav class="sb-nav">
    <div class="nav-sec">
      <div class="nav-lbl">Overview</div>
      <div class="nav-item active" onclick="go('overview',this)"><span class="ic">📊</span>Dashboard</div>
      <div class="nav-item" onclick="go('models',this)"><span class="ic">🧠</span>Model Status</div>
    </div>
    <div class="nav-sec">
      <div class="nav-lbl">ML Modules</div>
      <div class="nav-item" onclick="go('assess',this)"><span class="ic">🏥</span>Full Assessment</div>
      <div class="nav-item" onclick="go('classify',this)"><span class="ic">🔬</span>Sepsis Risk</div>
      <div class="nav-item" onclick="go('regress',this)"><span class="ic">📈</span>HR Forecast</div>
      <div class="nav-item" onclick="go('cluster',this)"><span class="ic">🎯</span>Risk Cluster</div>
      <div class="nav-item" onclick="go('forecast',this)"><span class="ic">⏱️</span>Vitals Forecast</div>
      <div class="nav-item" onclick="go('recommend',this)"><span class="ic">💊</span>Treatment Suggest</div>
      <div class="nav-item" onclick="go('pca',this)"><span class="ic">🔭</span>PCA Reduction</div>
      <div class="nav-item" onclick="go('assoc',this)"><span class="ic">🔗</span>Association Rules</div>
    </div>
    <div class="nav-sec">
      <div class="nav-lbl">System</div>
      <div class="nav-item" onclick="go('api',this)"><span class="ic">⚙️</span>API Reference</div>
    </div>
  </nav>
  <div class="sb-foot"><span class="dot"></span>All Systems Online<br/><span style="margin-left:12px">Updated: <span id="sync">—</span></span></div>
</aside>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div>
      <div class="pg-title" id="pgTitle">Clinical Dashboard</div>
      <div class="pg-sub" id="pgSub">Real-time sepsis intelligence & ICU monitoring</div>
    </div>
    <div class="tb-right">
      <div class="clock" id="clk">—</div>
      <div class="avatar">DR</div>
    </div>
  </div>

  <div class="content">

    <!-- OVERVIEW -->
    <div id="overview" class="sec active">
      <div class="info-bar"><span>🏥</span><span><strong>VitalWatch</strong> — AI-powered sepsis early warning for ICU patients. Enter patient vitals to get real-time risk assessment from 6 ML models.</span></div>
      <div class="stat-grid">
        <div class="stat ok"><div class="stat-ico">🟢</div><div class="stat-val">6</div><div class="stat-lbl">Active ML Models</div><div class="stat-note">All operational</div></div>
        <div class="stat"><div class="stat-ico">🔬</div><div class="stat-val">7</div><div class="stat-lbl">ML Task Types</div><div class="stat-note">Classification · Regression · Clustering…</div></div>
        <div class="stat warn"><div class="stat-ico">⚡</div><div class="stat-val">&lt;100ms</div><div class="stat-lbl">Inference Latency</div><div class="stat-note">FastAPI optimized</div></div>
        <div class="stat"><div class="stat-ico">🐳</div><div class="stat-val">CI/CD</div><div class="stat-lbl">Docker + Prefect</div><div class="stat-note">MLOps pipeline active</div></div>
      </div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Pipeline Flow</div><div class="card-tag">MLOps</div></div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:12.5px">
            <span style="background:rgba(0,200,240,.12);border:1px solid var(--border);border-radius:6px;padding:6px 11px;color:var(--cyan);font-weight:600">PostgreSQL</span><span style="color:var(--dim)">→</span>
            <span style="background:rgba(0,80,180,.15);border:1px solid var(--border);border-radius:6px;padding:6px 11px;color:#7eb7ff;font-weight:600">Feature Eng.</span><span style="color:var(--dim)">→</span>
            <span style="background:rgba(46,196,182,.1);border:1px solid var(--border);border-radius:6px;padding:6px 11px;color:var(--ok);font-weight:600">Prefect Train</span><span style="color:var(--dim)">→</span>
            <span style="background:rgba(244,162,97,.1);border:1px solid var(--border);border-radius:6px;padding:6px 11px;color:var(--warn);font-weight:600">FastAPI Serve</span><span style="color:var(--dim)">→</span>
            <span style="background:rgba(0,200,240,.12);border:1px solid var(--border);border-radius:6px;padding:6px 11px;color:var(--cyan);font-weight:600">Dashboard</span>
          </div>
          <div class="div"></div>
          <div style="font-size:12px;color:var(--dim);line-height:1.8">
            Orchestrated by <strong style="color:var(--white)">Prefect</strong> ·
            Tested with <strong style="color:var(--white)">DeepChecks</strong> ·
            CI/CD via <strong style="color:var(--white)">GitHub Actions</strong> ·
            Containerized with <strong style="color:var(--white)">Docker</strong>
          </div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">Quick Actions</div><div class="card-tag">Live</div></div>
          <div style="display:flex;flex-direction:column;gap:10px">
            <button class="btn btn-p" onclick="go('assess',document.querySelector('[onclick*=assess]'))">🏥 Run Full Patient Assessment</button>
            <button class="btn btn-p" onclick="go('forecast',document.querySelector('[onclick*=forecast]'))">⏱️ HR Time Series Forecast</button>
            <button class="btn btn-o" onclick="go('assoc',document.querySelector('[onclick*=assoc]'))">🔗 Load Association Rules</button>
            <button class="btn btn-o" onclick="testHealth()">🔎 Check System Health</button>
          </div>
          <div id="health_out" class="result" style="margin-top:10px">
            <pre id="health_json" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);white-space:pre-wrap"></pre>
          </div>
        </div>
      </div>
    </div>

    <!-- MODELS -->
    <div id="models" class="sec">
      <div class="three-col">
        <div class="mc"><div class="mc-name">Best Classifier</div><div class="mc-type">Sepsis Risk Classification</div><div class="mmet"><span class="ml">Medical Score</span><span class="mv" id="ms_clf">—</span></div><div class="mbar"><div class="mfill" id="mb_clf" style="width:0%"></div></div><div class="mmet"><span class="ml">AUROC</span><span class="mv" id="ms_auroc">—</span></div><div class="mbar"><div class="mfill" id="mb_auroc" style="width:0%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
        <div class="mc"><div class="mc-name">Best Regressor</div><div class="mc-type">Next-Hour HR Prediction</div><div class="mmet"><span class="ml">RMSE</span><span class="mv" id="ms_reg">—</span></div><div class="mbar"><div class="mfill" id="mb_reg" style="width:80%"></div></div><div class="mmet"><span class="ml">Target</span><span class="mv">hr_next_hour</span></div><div class="mbar"><div class="mfill" style="width:85%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
        <div class="mc"><div class="mc-name">Time Series Model</div><div class="mc-type">6-Hour HR Forecasting</div><div class="mmet"><span class="ml">Forecast Horizon</span><span class="mv">6 hours</span></div><div class="mbar"><div class="mfill" style="width:78%"></div></div><div class="mmet"><span class="ml">Features</span><span class="mv">Rolling Window</span></div><div class="mbar"><div class="mfill" style="width:70%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
        <div class="mc"><div class="mc-name">KMeans Clustering</div><div class="mc-type">Patient Risk Segmentation (4 clusters)</div><div class="mmet"><span class="ml">Clusters</span><span class="mv">4</span></div><div class="mbar"><div class="mfill" style="width:74%"></div></div><div class="mmet"><span class="ml">Sample Size</span><span class="mv">50,000</span></div><div class="mbar"><div class="mfill" style="width:80%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
        <div class="mc"><div class="mc-name">KNN Recommender</div><div class="mc-type">Similar Patient Treatment</div><div class="mmet"><span class="ml">Algorithm</span><span class="mv">KNN</span></div><div class="mbar"><div class="mfill" style="width:88%"></div></div><div class="mmet"><span class="ml">Features</span><span class="mv">12</span></div><div class="mbar"><div class="mfill" style="width:84%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
        <div class="mc"><div class="mc-name">PCA Reducer</div><div class="mc-type">Dimensionality Reduction</div><div class="mmet"><span class="ml">Components</span><span class="mv" id="ms_pca">2</span></div><div class="mbar"><div class="mfill" id="mb_pca" style="width:0%"></div></div><div class="mmet"><span class="ml">Input Features</span><span class="mv">28</span></div><div class="mbar"><div class="mfill" style="width:90%"></div></div><span class="badge badge-low" style="margin-top:8px">✓ Loaded</span></div>
      </div>
      <button class="btn btn-p btn-sm" onclick="loadManifest()" style="margin-top:4px">📋 Load Live Metrics from Manifest</button>
    </div>

    <!-- FULL ASSESSMENT -->
    <div id="assess" class="sec">
      <div class="info-bar"><span>🏥</span><span>Enter all patient vitals for a complete assessment using all 4 models simultaneously.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Patient Vitals</div><div class="card-tag">28 Features</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Heart Rate (bpm)</label><input class="fi" type="number" id="a_hr" value="90"/></div>
            <div class="fg"><label class="fl">O₂ Saturation (%)</label><input class="fi" type="number" id="a_o2" value="94" step="0.1"/></div>
            <div class="fg"><label class="fl">Temperature (°C)</label><input class="fi" type="number" id="a_temp" value="38.2" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP (mmHg)</label><input class="fi" type="number" id="a_sbp" value="105"/></div>
            <div class="fg"><label class="fl">MAP (mmHg)</label><input class="fi" type="number" id="a_map" value="70"/></div>
            <div class="fg"><label class="fl">DBP (mmHg)</label><input class="fi" type="number" id="a_dbp" value="60"/></div>
            <div class="fg"><label class="fl">Resp Rate (/min)</label><input class="fi" type="number" id="a_resp" value="22"/></div>
            <div class="fg"><label class="fl">Age</label><input class="fi" type="number" id="a_age" value="65"/></div>
            <div class="fg"><label class="fl">Gender (0=F, 1=M)</label><input class="fi" type="number" id="a_gender" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Hours in ICU</label><input class="fi" type="number" id="a_icu" value="6"/></div>
            <div class="fg"><label class="fl">HR Mean 3h</label><input class="fi" type="number" id="a_hrm3" value="88"/></div>
            <div class="fg"><label class="fl">HR Mean 6h</label><input class="fi" type="number" id="a_hrm6" value="85"/></div>
            <div class="fg"><label class="fl">SBP Mean 3h</label><input class="fi" type="number" id="a_sbpm3" value="108"/></div>
            <div class="fg"><label class="fl">MAP Mean 3h</label><input class="fi" type="number" id="a_mapm3" value="72"/></div>
            <div class="fg"><label class="fl">Resp Mean 3h</label><input class="fi" type="number" id="a_respm3" value="21"/></div>
            <div class="fg"><label class="fl">HR Trend</label><input class="fi" type="number" id="a_hrt" value="2" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP Trend</label><input class="fi" type="number" id="a_sbpt" value="-3" step="0.1"/></div>
            <div class="fg"><label class="fl">Temp Trend</label><input class="fi" type="number" id="a_tempt" value="0.2" step="0.1"/></div>
            <div class="fg"><label class="fl">Resp Trend</label><input class="fi" type="number" id="a_respt" value="1" step="0.1"/></div>
            <div class="fg"><label class="fl">HR Std 3h</label><input class="fi" type="number" id="a_hrstd" value="4" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP Std 3h</label><input class="fi" type="number" id="a_sbpstd" value="5" step="0.1"/></div>
            <div class="fg"><label class="fl">Flag: Tachycardia</label><input class="fi" type="number" id="a_tachy" value="0" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Flag: Hypoxia</label><input class="fi" type="number" id="a_hypox" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Flag: Fever</label><input class="fi" type="number" id="a_fever" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Flag: Hypothermia</label><input class="fi" type="number" id="a_hypot" value="0" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Flag: Low BP</label><input class="fi" type="number" id="a_lbp" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Flag: Tachypnea</label><input class="fi" type="number" id="a_tachyp" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">qSOFA Score (0-3)</label><input class="fi" type="number" id="a_qsofa" value="2" min="0" max="3"/></div>
          </div>
          <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
            <button class="btn btn-p" onclick="runAssess()">🏥 Run Full Assessment</button>
            <button class="btn btn-o" onclick="loadDemo()">Load High-Risk Demo</button>
          </div>
          <div class="loader" id="a_load"><div class="spin"></div>Analyzing all models…</div>
        </div>
        <div>
          <!-- Sepsis result -->
          <div class="card" id="res_sep" style="display:none">
            <div class="card-hd"><div class="card-title">Sepsis Prediction</div><div class="card-tag" id="res_level_tag">—</div></div>
            <div style="text-align:center;padding:10px 0">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:48px;font-weight:700;line-height:1" id="res_prob">—</div>
              <div style="font-size:11.5px;color:var(--dim);margin-top:3px">Sepsis Probability</div>
              <div id="res_badge" style="margin-top:10px"></div>
            </div>
            <div class="div"></div>
            <div id="res_action_box"></div>
            <div class="flags" id="res_flags"></div>
          </div>
          <!-- HR + Cluster + PCA -->
          <div class="card" id="res_extra" style="display:none">
            <div class="card-hd"><div class="card-title">Model Results</div><div class="card-tag">All Models</div></div>
            <div class="rrow"><span class="rk">Next-Hour HR</span><span class="rv c" id="res_hr">—</span></div>
            <div class="rrow"><span class="rk">HR Trend</span><span class="rv" id="res_trend">—</span></div>
            <div class="rrow"><span class="rk">Risk Cluster</span><span class="rv c" id="res_cluster">—</span></div>
            <div class="rrow"><span class="rk">PCA Zone</span><span class="rv" id="res_pca">—</span></div>
            <div class="rrow"><span class="rk">PC1 / PC2</span><span class="rv" id="res_coords">—</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- SEPSIS CLASSIFY -->
    <div id="classify" class="sec">
      <div class="info-bar"><span>🔬</span><span>Quick sepsis probability check using just the core vitals.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Quick Vitals Input</div><div class="card-tag">Classification</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Heart Rate</label><input class="fi" type="number" id="cl_hr" value="95"/></div>
            <div class="fg"><label class="fl">O₂ Saturation</label><input class="fi" type="number" id="cl_o2" value="93" step="0.1"/></div>
            <div class="fg"><label class="fl">Temperature (°C)</label><input class="fi" type="number" id="cl_temp" value="38.5" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP (mmHg)</label><input class="fi" type="number" id="cl_sbp" value="100"/></div>
            <div class="fg"><label class="fl">MAP (mmHg)</label><input class="fi" type="number" id="cl_map" value="65"/></div>
            <div class="fg"><label class="fl">Resp Rate</label><input class="fi" type="number" id="cl_resp" value="24"/></div>
            <div class="fg"><label class="fl">Age</label><input class="fi" type="number" id="cl_age" value="68"/></div>
            <div class="fg"><label class="fl">qSOFA (0-3)</label><input class="fi" type="number" id="cl_qsofa" value="2" min="0" max="3"/></div>
          </div>
          <div style="margin-top:14px;display:flex;gap:10px">
            <button class="btn btn-p" onclick="runClassify()">🔬 Classify</button>
          </div>
          <div class="loader" id="cl_load"><div class="spin"></div>Running…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">Result</div><div class="card-tag">Output</div></div>
          <div id="cl_res" class="result">
            <div class="rrow"><span class="rk">Probability</span><span class="rv c" id="cl_prob">—</span></div>
            <div class="rrow"><span class="rk">Risk Level</span><span class="rv" id="cl_level">—</span></div>
            <div class="rrow"><span class="rk">Prediction</span><span class="rv" id="cl_pred">—</span></div>
            <div id="cl_act"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- HR REGRESSION -->
    <div id="regress" class="sec">
      <div class="info-bar"><span>📈</span><span>Predict what the patient's heart rate will be in the next hour.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Current Vitals</div><div class="card-tag">Regression</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Current HR (bpm)</label><input class="fi" type="number" id="rg_hr" value="88"/></div>
            <div class="fg"><label class="fl">O₂ Saturation</label><input class="fi" type="number" id="rg_o2" value="95" step="0.1"/></div>
            <div class="fg"><label class="fl">Temperature (°C)</label><input class="fi" type="number" id="rg_temp" value="37.8" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP (mmHg)</label><input class="fi" type="number" id="rg_sbp" value="112"/></div>
            <div class="fg"><label class="fl">MAP (mmHg)</label><input class="fi" type="number" id="rg_map" value="75"/></div>
            <div class="fg"><label class="fl">HR Trend</label><input class="fi" type="number" id="rg_hrt" value="3" step="0.1"/></div>
            <div class="fg"><label class="fl">Age</label><input class="fi" type="number" id="rg_age" value="55"/></div>
            <div class="fg"><label class="fl">Hours in ICU</label><input class="fi" type="number" id="rg_icu" value="4"/></div>
          </div>
          <div style="margin-top:14px">
            <button class="btn btn-p" onclick="runRegress()">📈 Predict Next HR</button>
          </div>
          <div class="loader" id="rg_load"><div class="spin"></div>Computing…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">HR Forecast</div><div class="card-tag">Output</div></div>
          <div id="rg_res" class="result">
            <div style="text-align:center;padding:12px 0">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:48px;font-weight:700;color:var(--cyan);line-height:1" id="rg_big">—</div>
              <div style="font-size:11.5px;color:var(--dim);margin-top:3px">Predicted HR (bpm)</div>
            </div>
            <div class="div"></div>
            <div class="rrow"><span class="rk">Current HR</span><span class="rv c" id="rg_cur">—</span></div>
            <div class="rrow"><span class="rk">Change</span><span class="rv" id="rg_change">—</span></div>
            <div class="rrow"><span class="rk">Trend</span><span class="rv" id="rg_trend">—</span></div>
            <div class="rrow"><span class="rk">Status</span><span class="rv" id="rg_status">—</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- CLUSTER -->
    <div id="cluster" class="sec">
      <div class="info-bar"><span>🎯</span><span>Assign the patient to one of 4 risk clusters based on their vitals.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Vitals Input</div><div class="card-tag">KMeans</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Heart Rate</label><input class="fi" type="number" id="ck_hr" value="98"/></div>
            <div class="fg"><label class="fl">O₂ Saturation</label><input class="fi" type="number" id="ck_o2" value="91"/></div>
            <div class="fg"><label class="fl">Temperature</label><input class="fi" type="number" id="ck_temp" value="38.7" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP</label><input class="fi" type="number" id="ck_sbp" value="98"/></div>
            <div class="fg"><label class="fl">MAP</label><input class="fi" type="number" id="ck_map" value="62"/></div>
            <div class="fg"><label class="fl">qSOFA</label><input class="fi" type="number" id="ck_qsofa" value="2" min="0" max="3"/></div>
          </div>
          <div style="margin-top:14px">
            <button class="btn btn-p" onclick="runCluster()">🎯 Assign Cluster</button>
          </div>
          <div class="loader" id="ck_load"><div class="spin"></div>Clustering…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">Cluster Result</div><div class="card-tag">Output</div></div>
          <div id="ck_res" class="result">
            <div class="rrow"><span class="rk">Cluster ID</span><span class="rv c" id="ck_id">—</span></div>
            <div class="rrow"><span class="rk">Risk Label</span><span class="rv" id="ck_label">—</span></div>
            <div id="ck_dists" style="margin-top:10px"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TIME SERIES FORECAST -->
    <div id="forecast" class="sec">
      <div class="info-bar"><span>⏱️</span><span>Enter the last 6 hours of a patient's heart rate to forecast the next 6 hours.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Last 6 Hours of HR</div><div class="card-tag">Time Series</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">HR 6 hours ago</label><input class="fi" type="number" id="ts0" value="82"/></div>
            <div class="fg"><label class="fl">HR 5 hours ago</label><input class="fi" type="number" id="ts1" value="84"/></div>
            <div class="fg"><label class="fl">HR 4 hours ago</label><input class="fi" type="number" id="ts2" value="87"/></div>
            <div class="fg"><label class="fl">HR 3 hours ago</label><input class="fi" type="number" id="ts3" value="91"/></div>
            <div class="fg"><label class="fl">HR 2 hours ago</label><input class="fi" type="number" id="ts4" value="95"/></div>
            <div class="fg"><label class="fl">HR 1 hour ago</label><input class="fi" type="number" id="ts5" value="99"/></div>
          </div>
          <div style="margin-top:14px;display:flex;gap:10px">
            <button class="btn btn-p" onclick="runForecast()">⏱️ Forecast Next 6 Hours</button>
            <button class="btn btn-o" onclick="loadRisingDemo()">Load Rising Demo</button>
          </div>
          <div class="loader" id="ts_load"><div class="spin"></div>Forecasting…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">6-Hour HR Forecast</div><div class="card-tag">Output</div></div>
          <div id="ts_res" class="result">
            <div class="rrow"><span class="rk">Alert</span><span class="rv" id="ts_alert">—</span></div>
            <div class="rrow"><span class="rk">Message</span><span class="rv" id="ts_msg" style="font-size:11.5px;white-space:normal;text-align:right;max-width:60%">—</span></div>
            <div class="fc-grid" id="ts_cells" style="margin-top:12px"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- RECOMMENDATION -->
    <div id="recommend" class="sec">
      <div class="info-bar"><span>💊</span><span>Find similar historical patients and get treatment recommendations based on KNN similarity.</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Patient Profile</div><div class="card-tag">KNN</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Avg HR</label><input class="fi" type="number" id="rc_hr" value="92"/></div>
            <div class="fg"><label class="fl">Avg Temperature</label><input class="fi" type="number" id="rc_temp" value="38.1" step="0.1"/></div>
            <div class="fg"><label class="fl">Avg O₂ Sat</label><input class="fi" type="number" id="rc_o2" value="93"/></div>
            <div class="fg"><label class="fl">Avg SBP</label><input class="fi" type="number" id="rc_sbp" value="108"/></div>
            <div class="fg"><label class="fl">Avg MAP</label><input class="fi" type="number" id="rc_map" value="70"/></div>
            <div class="fg"><label class="fl">Avg Resp Rate</label><input class="fi" type="number" id="rc_resp" value="20"/></div>
            <div class="fg"><label class="fl">Max qSOFA</label><input class="fi" type="number" id="rc_qsofa" value="2" min="0" max="3"/></div>
            <div class="fg"><label class="fl">Total ICU Hours</label><input class="fi" type="number" id="rc_icu" value="24"/></div>
            <div class="fg"><label class="fl">Had Tachycardia</label><input class="fi" type="number" id="rc_tachy" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Had Fever</label><input class="fi" type="number" id="rc_fever" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Had Low BP</label><input class="fi" type="number" id="rc_lbp" value="1" min="0" max="1"/></div>
            <div class="fg"><label class="fl">Had Hypoxia</label><input class="fi" type="number" id="rc_hypox" value="0" min="0" max="1"/></div>
          </div>
          <div style="margin-top:14px">
            <button class="btn btn-p" onclick="runRecommend()">💊 Find Similar Patients</button>
          </div>
          <div class="loader" id="rc_load"><div class="spin"></div>Searching…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">Similar Patients</div><div class="card-tag">Output</div></div>
          <div id="rc_res" class="result">
            <div class="rrow"><span class="rk">Risk Level</span><span class="rv" id="rc_level">—</span></div>
            <div class="rrow"><span class="rk">Est. Risk %</span><span class="rv c" id="rc_risk">—</span></div>
            <div class="rrow"><span class="rk">Sepsis in Similar</span><span class="rv" id="rc_sep">—</span></div>
            <div id="rc_act" style="margin-top:10px"></div>
            <div id="rc_patients" style="margin-top:12px;font-size:11.5px;color:var(--dim)"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- PCA -->
    <div id="pca" class="sec">
      <div class="info-bar"><span>🔭</span><span>See where the patient sits in the 2D health space (cardiovascular vs respiratory stress).</span></div>
      <div class="two-col">
        <div class="card">
          <div class="card-hd"><div class="card-title">Patient Vitals</div><div class="card-tag">PCA</div></div>
          <div class="form-grid">
            <div class="fg"><label class="fl">Heart Rate</label><input class="fi" type="number" id="pc_hr" value="95"/></div>
            <div class="fg"><label class="fl">O₂ Saturation</label><input class="fi" type="number" id="pc_o2" value="92"/></div>
            <div class="fg"><label class="fl">Temperature</label><input class="fi" type="number" id="pc_temp" value="38.8" step="0.1"/></div>
            <div class="fg"><label class="fl">SBP</label><input class="fi" type="number" id="pc_sbp" value="100"/></div>
            <div class="fg"><label class="fl">Resp Rate</label><input class="fi" type="number" id="pc_resp" value="24"/></div>
            <div class="fg"><label class="fl">MAP</label><input class="fi" type="number" id="pc_map" value="64"/></div>
          </div>
          <div style="margin-top:14px">
            <button class="btn btn-p" onclick="runPCA()">🔭 Reduce Dimensions</button>
          </div>
          <div class="loader" id="pc_load"><div class="spin"></div>Applying PCA…</div>
        </div>
        <div class="card">
          <div class="card-hd"><div class="card-title">PCA Result</div><div class="card-tag">Output</div></div>
          <div id="pc_res" class="result">
            <div class="rrow"><span class="rk">Health Zone</span><span class="rv" id="pc_zone">—</span></div>
            <div class="rrow"><span class="rk">PC1 (Cardiovascular)</span><span class="rv c" id="pc_1">—</span></div>
            <div class="rrow"><span class="rk">PC2 (Resp/Temp)</span><span class="rv c" id="pc_2">—</span></div>
            <div class="rrow"><span class="rk">Variance Explained</span><span class="rv g" id="pc_var">—</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ASSOCIATION RULES -->
    <div id="assoc" class="sec">
      <div class="card">
        <div class="card-hd"><div class="card-title">Medical Association Rules</div><div class="card-tag">Apriori</div></div>
        <button class="btn btn-p btn-sm" onclick="loadAssoc()">🔗 Load Rules</button>
        <div class="loader" id="as_load"><div class="spin"></div>Loading…</div>
        <div id="as_wrap" style="margin-top:16px;display:none;overflow-x:auto">
          <table class="tbl">
            <thead><tr><th>Antecedents</th><th>Consequents</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
            <tbody id="as_body"></tbody>
          </table>
        </div>
        <div id="as_note" style="margin-top:12px;font-size:12px;color:var(--dim)"></div>
      </div>
    </div>

    <!-- API -->
    <div id="api" class="sec">
      <div class="card">
        <div class="card-hd"><div class="card-title">API Endpoints</div><div class="card-tag">FastAPI</div></div>
        <table class="tbl">
          <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th><th>Test</th></tr></thead>
          <tbody>
            <tr><td style="color:var(--ok);font-family:monospace;font-weight:700">GET</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/health</code></td><td style="color:var(--dim)">System health check</td><td><button class="btn btn-o btn-sm" onclick="apiTest('/health')">Test</button></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/assess</code></td><td style="color:var(--dim)">Full patient assessment (all models)</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/classify</code></td><td style="color:var(--dim)">Sepsis risk classification</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/regress</code></td><td style="color:var(--dim)">Next-hour HR prediction</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/cluster</code></td><td style="color:var(--dim)">Patient cluster assignment</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/forecast</code></td><td style="color:var(--dim)">6-hour HR forecast</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/recommend</code></td><td style="color:var(--dim)">Treatment recommendation</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--warn);font-family:monospace;font-weight:700">POST</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/pca</code></td><td style="color:var(--dim)">PCA dimensionality reduction</td><td><a href="/api/docs" target="_blank"><button class="btn btn-o btn-sm">Docs</button></a></td></tr>
            <tr><td style="color:var(--ok);font-family:monospace;font-weight:700">GET</td><td><code style="color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:11.5px">/api/association-rules</code></td><td style="color:var(--dim)">Medical association rules</td><td><button class="btn btn-o btn-sm" onclick="apiTest('/api/association-rules')">Test</button></td></tr>
          </tbody>
        </table>
        <div class="div"></div>
        <div style="display:flex;gap:10px">
          <a href="/api/docs" target="_blank"><button class="btn btn-p">📖 Swagger UI</button></a>
          <a href="/api/redoc" target="_blank"><button class="btn btn-o">📄 ReDoc</button></a>
        </div>
        <div id="api_res" class="result" style="margin-top:12px">
          <pre id="api_json" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);white-space:pre-wrap;word-break:break-all"></pre>
        </div>
      </div>
    </div>

  </div>
</div>
</div>

<script>
const titles={overview:['Clinical Dashboard','Real-time sepsis intelligence & ICU monitoring'],models:['Model Status','6 active ML models · Sepsis domain'],assess:['Full Assessment','All 4 models simultaneously'],classify:['Sepsis Risk','Quick sepsis probability check'],regress:['HR Forecast','Next-hour heart rate prediction'],cluster:['Risk Cluster','KMeans patient segmentation'],forecast:['Vitals Forecast','6-hour HR time series forecast'],recommend:['Treatment Suggestion','KNN similar patient matching'],pca:['PCA Reduction','2D health space visualization'],assoc:['Association Rules','Medical co-occurrence patterns'],api:['API Reference','FastAPI endpoints']};

function go(id,el){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if(el)el.classList.add('active');
  const t=titles[id]||[id,''];
  document.getElementById('pgTitle').textContent=t[0];
  document.getElementById('pgSub').textContent=t[1];
}

function tick(){
  const n=new Date();
  document.getElementById('clk').textContent=n.toLocaleTimeString('en-GB',{hour12:false});
  document.getElementById('sync').textContent=n.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'});
}
setInterval(tick,1000);tick();

function load(id,on){const e=document.getElementById(id);if(e)e.classList.toggle('show',on);}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw new Error('HTTP '+r.status+': '+await r.text());return r.json();}
async function get(url){const r=await fetch(url);if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}

function v(id){return parseFloat(document.getElementById(id)?.value)||0;}
function set(id,val,cls=''){const e=document.getElementById(id);if(e){e.textContent=val;e.className='rv '+(cls||'');}}

function getVitals(prefix){
  const map={hr:'hr',o2:'o2sat',temp:'temp',sbp:'sbp',map:'map',dbp:'dbp',resp:'resp',age:'age',gender:'gender',icu:'hours_in_icu',hrm3:'hr_mean_3h',hrm6:'hr_mean_6h',sbpm3:'sbp_mean_3h',mapm3:'map_mean_3h',respm3:'resp_mean_3h',hrt:'hr_trend',sbpt:'sbp_trend',tempt:'temp_trend',respt:'resp_trend',hrstd:'hr_std_3h',sbpstd:'sbp_std_3h',tachy:'flag_tachy',hypox:'flag_hypoxia',fever:'flag_fever',hypot:'flag_hypotemp',lbp:'flag_low_bp',tachyp:'flag_tachypnea',qsofa:'qsofa'};
  const o={};
  Object.entries(map).forEach(([k,fk])=>{const el=document.getElementById(prefix+'_'+k);if(el)o[fk]=parseFloat(el.value)||0;});
  return o;
}

// ── DEMO ──
function loadDemo(){
  const d={a_hr:102,a_o2:90,a_temp:39.1,a_sbp:88,a_map:58,a_dbp:48,a_resp:26,a_age:72,a_gender:1,a_icu:12,a_hrm3:98,a_hrm6:94,a_sbpm3:92,a_mapm3:62,a_respm3:24,a_hrt:4,a_sbpt:-5,a_tempt:0.4,a_respt:2,a_hrstd:6,a_sbpstd:8,a_tachy:1,a_hypox:1,a_fever:1,a_hypot:0,a_lbp:1,a_tachyp:1,a_qsofa:3};
  Object.entries(d).forEach(([k,v])=>{const e=document.getElementById(k);if(e)e.value=v;});
}
function loadRisingDemo(){
  [90,93,96,100,104,108].forEach((v,i)=>{const e=document.getElementById('ts'+i);if(e)e.value=v;});
}

// ── FULL ASSESSMENT ──
async function runAssess(){
  load('a_load',true);
  document.getElementById('res_sep').style.display='none';
  document.getElementById('res_extra').style.display='none';
  try{
    const vitals=getVitals('a');
    const d=await post('/api/assess',vitals);
    const sep=d.sepsis;
    const prob=Math.round(sep.probability*100);
    const lvl=sep.risk_level;
    const cls=lvl==='HIGH RISK'?'d':lvl==='MODERATE RISK'?'w':'g';
    const bcls=lvl==='HIGH RISK'?'badge-high':lvl==='MODERATE RISK'?'badge-mod':'badge-low';

    document.getElementById('res_prob').textContent=prob+'%';
    document.getElementById('res_prob').style.color=lvl==='HIGH RISK'?'var(--danger)':lvl==='MODERATE RISK'?'var(--warn)':'var(--ok)';
    document.getElementById('res_level_tag').textContent=lvl;
    document.getElementById('res_badge').innerHTML=`<span class="badge ${bcls}">${lvl}</span>`;

    const actCls=lvl==='HIGH RISK'?'high':lvl==='MODERATE RISK'?'mod':'';
    document.getElementById('res_action_box').innerHTML=`<div class="action-box ${actCls}">${sep.action}</div>`;

    const flagsHtml=(sep.flags||[]).map(f=>`<span class="flag">${f}</span>`).join('');
    document.getElementById('res_flags').innerHTML=flagsHtml;

    document.getElementById('res_sep').style.display='block';

    const hr=d.hr_forecast;
    const trendIcon=hr.trend==='rising'?'↑ rising':hr.trend==='falling'?'↓ falling':'→ stable';
    const trendCls=hr.trend==='rising'?'d':hr.trend==='falling'?'g':'';
    set('res_hr',hr.next_hour+' bpm','c');
    set('res_trend',trendIcon,trendCls);
    set('res_cluster',d.cluster.name,'c');

    const zone=d.pca.pc1>1.5||d.pca.pc2>1.5?'🔴 RED ZONE':d.pca.pc1>0.5||d.pca.pc2>0.5?'🟡 YELLOW ZONE':'🟢 GREEN ZONE';
    set('res_pca',zone,d.pca.pc1>1.5?'d':d.pca.pc1>0.5?'w':'g');
    set('res_coords',`PC1: ${d.pca.pc1} | PC2: ${d.pca.pc2}`,'');

    document.getElementById('res_extra').style.display='block';
  }catch(e){alert('Error: '+e.message);}
  finally{load('a_load',false);}
}

// ── CLASSIFY ──
async function runClassify(){
  load('cl_load',true);
  document.getElementById('cl_res').classList.remove('show');
  try{
    const body={hr:v('cl_hr'),o2sat:v('cl_o2'),temp:v('cl_temp'),sbp:v('cl_sbp'),map:v('cl_map'),dbp:v('cl_map')-10,resp:v('cl_resp'),age:v('cl_age'),gender:1,hr_mean_3h:v('cl_hr'),hr_mean_6h:v('cl_hr'),sbp_mean_3h:v('cl_sbp'),map_mean_3h:v('cl_map'),resp_mean_3h:v('cl_resp'),hr_trend:0,sbp_trend:0,temp_trend:0,resp_trend:0,hr_std_3h:2,sbp_std_3h:3,flag_tachy:v('cl_hr')>100?1:0,flag_hypoxia:v('cl_o2')<92?1:0,flag_fever:v('cl_temp')>38.3?1:0,flag_hypotemp:0,flag_low_bp:v('cl_map')<65?1:0,flag_tachypnea:v('cl_resp')>22?1:0,qsofa:v('cl_qsofa'),hours_in_icu:6};
    const d=await post('/api/classify',body);
    const lvl=d.risk_level;
    const cls=lvl==='HIGH RISK'?'d':lvl==='MODERATE RISK'?'w':'g';
    const bcls=lvl==='HIGH RISK'?'badge-high':lvl==='MODERATE RISK'?'badge-mod':'badge-low';
    set('cl_prob',Math.round(d.probability*100)+'%','c');
    document.getElementById('cl_level').innerHTML=`<span class="badge ${bcls}">${lvl}</span>`;
    set('cl_pred',d.prediction===1?'Sepsis Positive':'Sepsis Negative',d.prediction===1?'d':'g');
    const actCls=lvl==='HIGH RISK'?'high':lvl==='MODERATE RISK'?'mod':'';
    document.getElementById('cl_act').innerHTML=`<div class="action-box ${actCls}" style="margin-top:10px">${d.action}</div>`;
    document.getElementById('cl_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('cl_load',false);}
}

// ── REGRESSION ──
async function runRegress(){
  load('rg_load',true);
  document.getElementById('rg_res').classList.remove('show');
  try{
    const body={hr:v('rg_hr'),o2sat:v('rg_o2'),temp:v('rg_temp'),sbp:v('rg_sbp'),map:v('rg_map'),dbp:v('rg_map')-10,resp:16,age:v('rg_age'),gender:1,hr_mean_3h:v('rg_hr'),hr_mean_6h:v('rg_hr'),sbp_mean_3h:v('rg_sbp'),map_mean_3h:v('rg_map'),resp_mean_3h:16,hr_trend:v('rg_hrt'),sbp_trend:0,temp_trend:0,resp_trend:0,hr_std_3h:3,sbp_std_3h:4,flag_tachy:v('rg_hr')>100?1:0,flag_hypoxia:v('rg_o2')<92?1:0,flag_fever:v('rg_temp')>38.3?1:0,flag_hypotemp:0,flag_low_bp:v('rg_map')<65?1:0,flag_tachypnea:0,qsofa:0,hours_in_icu:v('rg_icu')};
    const d=await post('/api/regress',body);
    document.getElementById('rg_big').textContent=d.predicted_hr;
    document.getElementById('rg_big').style.color=d.predicted_hr>120?'var(--danger)':d.predicted_hr>100?'var(--warn)':'var(--cyan)';
    set('rg_cur',d.current_hr+' bpm','c');
    set('rg_change',(d.change>0?'+':'')+d.change+' bpm',d.change>3?'d':d.change<-3?'g':'');
    set('rg_trend',d.trend,d.trend==='rising'?'d':d.trend==='falling'?'g':'');
    set('rg_status',d.status,d.status==='CRITICAL'?'d':d.status==='WARNING'?'w':'g');
    document.getElementById('rg_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('rg_load',false);}
}

// ── CLUSTER ──
async function runCluster(){
  load('ck_load',true);
  document.getElementById('ck_res').classList.remove('show');
  try{
    const body={hr:v('ck_hr'),o2sat:v('ck_o2'),temp:v('ck_temp'),sbp:v('ck_sbp'),map:v('ck_map'),dbp:v('ck_map')-10,resp:18,age:60,gender:1,hr_mean_3h:v('ck_hr'),hr_mean_6h:v('ck_hr'),sbp_mean_3h:v('ck_sbp'),map_mean_3h:v('ck_map'),resp_mean_3h:18,hr_trend:0,sbp_trend:0,temp_trend:0,resp_trend:0,hr_std_3h:3,sbp_std_3h:4,flag_tachy:v('ck_hr')>100?1:0,flag_hypoxia:v('ck_o2')<92?1:0,flag_fever:v('ck_temp')>38.3?1:0,flag_hypotemp:0,flag_low_bp:v('ck_map')<65?1:0,flag_tachypnea:0,qsofa:v('ck_qsofa'),hours_in_icu:6};
    const d=await post('/api/cluster',body);
    set('ck_id','Cluster '+d.cluster_id,'c');
    set('ck_label',d.risk_label,d.risk_label?.includes('Critical')?'d':d.risk_label?.includes('High')?'w':d.risk_label?.includes('Elevated')?'w':'g');
    const distHtml=Object.entries(d.all_distances||{}).map(([k,v])=>`<div class="rrow"><span class="rk">${k}</span><span class="rv" style="font-size:11px">${v}</span></div>`).join('');
    document.getElementById('ck_dists').innerHTML=distHtml;
    document.getElementById('ck_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('ck_load',false);}
}

// ── TIME SERIES ──
async function runForecast(){
  load('ts_load',true);
  document.getElementById('ts_res').classList.remove('show');
  try{
    const hrs=[0,1,2,3,4,5].map(i=>parseFloat(document.getElementById('ts'+i).value)||80);
    const d=await post('/api/forecast',{hr_last_6_hours:hrs,patient_id:'dashboard'});
    set('ts_alert',d.alert?'⚠️ DANGER':'✅ Safe',d.alert?'d':'g');
    set('ts_msg',d.alert_message,'');
    const cells=(d.forecast||[]).map(f=>`<div class="fc-cell ${f.alert?'alert':''}"><div class="fc-h">${f.hour}</div><div class="fc-v ${f.alert?'hot':''}">${f.hr}</div></div>`).join('');
    document.getElementById('ts_cells').innerHTML=cells;
    document.getElementById('ts_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('ts_load',false);}
}

// ── RECOMMEND ──
async function runRecommend(){
  load('rc_load',true);
  document.getElementById('rc_res').classList.remove('show');
  try{
    const body={avg_hr:v('rc_hr'),avg_temp:v('rc_temp'),avg_o2sat:v('rc_o2'),avg_sbp:v('rc_sbp'),avg_map:v('rc_map'),avg_resp:v('rc_resp'),max_qsofa:v('rc_qsofa'),had_tachy:v('rc_tachy'),had_fever:v('rc_fever'),had_low_bp:v('rc_lbp'),had_hypoxia:v('rc_hypox'),total_icu_hours:v('rc_icu')};
    const d=await post('/api/recommend',body);
    const lvl=d.risk_level;
    const bcls=lvl==='HIGH RISK'?'badge-high':lvl==='MODERATE RISK'?'badge-mod':'badge-low';
    document.getElementById('rc_level').innerHTML=`<span class="badge ${bcls}">${lvl}</span>`;
    set('rc_risk',d.estimated_risk_pct+'%',d.estimated_risk_pct>50?'d':d.estimated_risk_pct>25?'w':'g');
    set('rc_sep',d.sepsis_in_similar+' / '+(d.similar_patients?.length||10),'');
    const actCls=lvl==='HIGH RISK'?'high':lvl==='MODERATE RISK'?'mod':'';
    document.getElementById('rc_act').innerHTML=`<div class="action-box ${actCls}">${d.recommendation}</div>`;
    const pts=(d.similar_patients||[]).slice(0,5).map(p=>`<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(0,200,240,.05)"><span>Patient ${p.patient_id}</span><span style="color:${p.had_sepsis?'var(--danger)':'var(--ok)'}">${p.had_sepsis?'Sepsis':'No Sepsis'} — ${p.similarity}</span></div>`).join('');
    document.getElementById('rc_patients').innerHTML=pts;
    document.getElementById('rc_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('rc_load',false);}
}

// ── PCA ──
async function runPCA(){
  load('pc_load',true);
  document.getElementById('pc_res').classList.remove('show');
  try{
    const body={hr:v('pc_hr'),o2sat:v('pc_o2'),temp:v('pc_temp'),sbp:v('pc_sbp'),map:v('pc_map'),dbp:v('pc_map')-10,resp:v('pc_resp'),age:60,gender:1,hr_mean_3h:v('pc_hr'),hr_mean_6h:v('pc_hr'),sbp_mean_3h:v('pc_sbp'),map_mean_3h:v('pc_map'),resp_mean_3h:v('pc_resp'),hr_trend:0,sbp_trend:0,temp_trend:0,resp_trend:0,hr_std_3h:3,sbp_std_3h:4,flag_tachy:v('pc_hr')>100?1:0,flag_hypoxia:v('pc_o2')<92?1:0,flag_fever:v('pc_temp')>38.3?1:0,flag_hypotemp:0,flag_low_bp:v('pc_map')<65?1:0,flag_tachypnea:v('pc_resp')>22?1:0,qsofa:0,hours_in_icu:6};
    const d=await post('/api/pca',body);
    const zoneCls=d.zone?.includes('RED')?'d':d.zone?.includes('YELLOW')?'w':'g';
    const zoneIcon=d.zone?.includes('RED')?'🔴':d.zone?.includes('YELLOW')?'🟡':'🟢';
    set('pc_zone',zoneIcon+' '+d.zone,zoneCls);
    set('pc_1',d.pc1,'c');
    set('pc_2',d.pc2,'c');
    const varArr=d.explained_variance||[];
    set('pc_var',varArr.length>0?varArr.map(v=>(v*100).toFixed(1)+'%').join(' | '):'—','g');
    document.getElementById('pc_res').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
  finally{load('pc_load',false);}
}

// ── ASSOC ──
async function loadAssoc(){
  load('as_load',true);
  try{
    const d=await get('/api/association-rules');
    const rules=d.top_rules||[];
    const tbody=document.getElementById('as_body');
    tbody.innerHTML=rules.length>0?rules.map(r=>`<tr><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${JSON.stringify(r.antecedents||'—')}</td><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${JSON.stringify(r.consequents||'—')}</td><td>${(+r.support).toFixed(3)}</td><td style="color:var(--cyan)">${(+r.confidence).toFixed(3)}</td><td style="color:var(--ok)">${(+r.lift).toFixed(2)}</td></tr>`).join(''):`<tr><td colspan="5" style="color:var(--dim);text-align:center;padding:18px">${d.note||'No sepsis-specific rules (below support threshold — expected)'}</td></tr>`;
    document.getElementById('as_note').textContent=d.note||'';
    document.getElementById('as_wrap').style.display='block';
  }catch(e){alert('Error: '+e.message);}
  finally{load('as_load',false);}
}

// ── HEALTH CHECK ──
async function testHealth(){
  try{
    const d=await get('/health');
    document.getElementById('health_json').textContent=JSON.stringify(d,null,2);
    document.getElementById('health_out').classList.add('show');
  }catch(e){alert('Error: '+e.message);}
}

// ── API TEST ──
async function apiTest(url){
  try{
    const d=await get(url);
    document.getElementById('api_json').textContent=JSON.stringify(d,null,2).slice(0,1000);
    document.getElementById('api_res').classList.add('show');
    go('api',document.querySelector('[onclick*=api]'));
  }catch(e){alert('Error: '+e.message);}
}

// ── LOAD MANIFEST ──
async function loadManifest(){
  try{
    const d=await get('/api/manifest');
    if(d.classification){
      const s=d.classification.best_medical_score||0;
      document.getElementById('ms_clf').textContent=s;
      document.getElementById('mb_clf').style.width=Math.round(s*100)+'%';
      const all=d.classification.all_results||{};
      const bestName=d.classification.best_model;
      const auroc=all[bestName]?.AUROC||0;
      document.getElementById('ms_auroc').textContent=auroc;
      document.getElementById('mb_auroc').style.width=Math.round(auroc*100)+'%';
    }
    if(d.regression){
      document.getElementById('ms_reg').textContent=d.regression.best_rmse||'—';
    }
    if(d.dimensionality_reduction){
      const total=d.dimensionality_reduction.variance_explained?.total||0;
      document.getElementById('ms_pca').textContent='2 ('+Math.round(total*100)+'% var)';
      document.getElementById('mb_pca').style.width=Math.round(total*100)+'%';
    }
  }catch(e){console.warn('Manifest load failed:',e.message);}
}

// Auto-load manifest
loadManifest();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
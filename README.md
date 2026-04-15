# VitalWatch: Real-time Sepsis Prediction System 🏥

## 📌 Project Overview
VitalWatch is an AI-driven monitoring system designed to predict the onset of sepsis in ICU patients. By analyzing hourly physiological vitals (Heart Rate, Temp, O2Sat), the system provides early warnings to clinical staff.

## 🛠 Tech Stack
- **Language:** Python 3.10+
- **Backend:** FastAPI (Uvicorn)
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Version Control:** Git & GitHub
- **Environment:** Ubuntu/Windows Virtual Environment (venv)

## 📂 Project Milestone Log
### Phase 1: Environment & Data Architecture 
- **Repository Setup:** Established a standardized directory structure.
- **Data Ingestion:** Sourced ~40,000 clinical records in `.psv` format.
- **Git Strategy:** Implemented a `.gitignore` policy to prevent repository bloating by excluding large-scale datasets.
- **API Foundation:** Developed base schemas for Patient and Vital signs using Pydantic.

## 🚀 Challenges Overcome
- **Data Mirroring:** Resolved PhysioNet 403 access issues by pivoting to a verified Kaggle data mirror.
- **Merge Conflicts:** Handled initial Git synchronization conflicts during the repository initialization phase.

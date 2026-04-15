from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import models, schemas, database

app = FastAPI()

# Create tables
models.Base.metadata.create_all(bind=database.engine)

# Dependency to get the database connection
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# NEW: This "Post" endpoint saves a patient record
@app.post("/add-patient/")
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    # Create the record object
    new_record = models.PatientRecord(
        heart_rate=patient.heart_rate,
        temp=patient.temp,
        sepsis_risk="Pending Calculation", 
        patient_name=patient.patient_name # We will add the AI model here later!
    )
    # Save it to PostgreSQL
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"message": "Patient data saved successfully!", "id": new_record.id}
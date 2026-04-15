from pydantic import BaseModel

# This defines what the doctor sends to the API
class PatientCreate(BaseModel):
    heart_rate: float
    temp: float
    patient_name: str
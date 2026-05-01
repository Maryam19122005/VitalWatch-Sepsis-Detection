from sqlalchemy import Column, Integer, Float, String
from database import Base

class PatientRecord(Base):
    __tablename__ = "sepsis_logs"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String) # <--- MAKE SURE THIS IS HERE
    heart_rate = Column(Float)
    temp = Column(Float)
    sepsis_risk = Column(String)
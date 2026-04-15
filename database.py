from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# The 'address' of your database
# Format: postgresql://username:password@localhost:port/database_name
DATABASE_URL = "postgresql://postgres:20051219@localhost:5432/vitalwatch_db"

# This engine is the actual 'pipe' connecting Python to PostgreSQL
engine = create_engine(DATABASE_URL)

# This allows us to talk to the database in 'sessions' (like a phone call)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is the base class that all our future tables will use
Base = declarative_base()
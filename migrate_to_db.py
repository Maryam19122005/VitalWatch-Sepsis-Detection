import pandas as pd
import glob
import os
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:1234@localhost:5432/vitalwatch_db')

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS patient_vitals CASCADE"))
    conn.commit()
    print("Starting fresh with correct columns...")

all_files = glob.glob('data/processed/*.psv')
print(f"Starting migration of {len(all_files)} files...")

for i, file_path in enumerate(all_files):
    try:
        df = pd.read_csv(file_path, sep=',')  # ← FIXED HERE
        
        if df.empty:
            continue
            
        df['patient_id'] = os.path.basename(file_path).replace('.psv', '')
        df.to_sql('patient_vitals', engine, if_exists='append', index=False)

    except pd.errors.EmptyDataError:
        continue
    except Exception as e:
        print(f"Skipped {os.path.basename(file_path)} due to: {e}")
        continue

    if (i + 1) % 500 == 0:
        print(f"Progress: {i+1} / {len(all_files)} files migrated.")

print("SUCCESS! Your data is fully loaded.")
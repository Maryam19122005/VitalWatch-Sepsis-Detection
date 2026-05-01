import pandas as pd
import glob
import os

if not os.path.exists('data/processed'):
    os.makedirs('data/processed')

# Columns that should NEVER be filled with 0
# These are lab values that are only taken occasionally
LAB_COLUMNS = [
    'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
    'PTT', 'WBC', 'Fibrinogen', 'Platelets'
]

# Vital signs that are measured every hour - forward fill is fine
VITAL_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']

all_files = glob.glob('data/raw/training/*.psv')
print(f"Found {len(all_files)} files to clean.")

for i, file_path in enumerate(all_files):
    file_name = os.path.basename(file_path)
    
    df = pd.read_csv(file_path, sep='|')
    
    # Step 1: Forward fill then backward fill ONLY vital signs
    # (these are hourly readings so interpolation makes sense)
    df[VITAL_COLUMNS] = df[VITAL_COLUMNS].ffill().bfill()
    
    # Step 2: For lab values, ONLY forward fill within same patient
    # If still missing, leave as NaN — do NOT fill with 0
    df[LAB_COLUMNS] = df[LAB_COLUMNS].ffill()
    
    # Step 3: For any remaining NaN in vitals only (edge case), use column median
    for col in VITAL_COLUMNS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Step 4: Save
    output_path = f'data/processed/{file_name}'
    df.to_csv(output_path, index=False)
    
    if (i + 1) % 2000 == 0:
        print(f"Progress: {i+1} / {len(all_files)} files cleaned.")

print("Done! All files cleaned correctly.")